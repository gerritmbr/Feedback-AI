import csv
import json
from typing import List, Dict, Tuple, Optional, Set
import anthropic # This import is not directly used in the provided snippet but is part of the original context
from dataclasses import dataclass
import time
import os
from pathlib import Path
from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.blocks.system import Secret
import httpx
import asyncio
import time
from datetime import datetime # Added for dynamic file naming
import re 

@dataclass
class QAPair:
    question: str
    answer: str
    reason: str
    transcript_id: str
    content_category: str

# Configuration
class PipelineConfig:
    LLM_API_URL = "https://api.anthropic.com/v1/messages"
    # LLM_MODEL = "claude-3-5-sonnet-20241022"  # Updated to available model
    LLM_MODEL = "claude-opus-4-20250514"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.1
    RATE_LIMIT_DELAY = 1  # seconds between API calls (for cleaning answers)
    RATE_LIMIT_DELAY_LONG = 5  # seconds between API calls (used for more complex tasks like flattening)
    BATCH_SIZE = 10 # General batch size, adjusted per task as needed


@task(name="import-qa-data")
def import_qa_pairs_from_csv(filename: str) -> List[QAPair]:
    """Import QA pairs from CSV with elegant error handling and default values."""
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [
            QAPair(
                question=row['question'],
                answer=row['answer'],
                reason=row['reason'],
                transcript_id=row['transcript_id'],
                content_category=row.get('content_category', '')
            )
            for row in reader
        ]


# Utility function to load processed transcript IDs from a file or database
@task(name="load-processed-ids")
def load_processed_transcript_ids(file_path: str) -> Set[str]:
    """Load a set of already processed transcript IDs from a file."""
    logger = get_run_logger()
    try:
        with open(file_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        logger.info(f"No processed transcript IDs file found at {file_path}")
        return set()

@task(name="safe-processed-ids")
def save_processed_transcript_ids(transcript_ids: Set[str], file_path: str):
    """Save processed transcript IDs to a file."""
    logger = get_run_logger()
    try:
        with open(file_path, 'w') as f:
            for tid in sorted(transcript_ids):
                f.write(f"{tid}\n")
        logger.info(f"Saved {len(transcript_ids)} processed transcript IDs to {file_path}")
    except Exception as e:
        logger.error(f"Error saving processed transcript IDs: {e}")

@task(name="clean_batch_answers", retries=3, retry_delay_seconds=2)
async def clean_answers_batch(qa_pairs: List[QAPair], api_key: str) -> Dict[str, str]:
    """Use Claude API to clean and summarize a batch of answers."""
    logger = get_run_logger()

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    # Filter out very short answers (3 words or less) - don't process them
    pairs_to_process = []
    skip_mapping = {}
    
    for qa_pair in qa_pairs:
        if len(qa_pair.answer.split()) <= 3:
            skip_mapping[qa_pair.answer] = qa_pair.answer  # Keep original
            logger.debug(f"Skipping short answer: '{qa_pair.answer}'")
        else:
            pairs_to_process.append(qa_pair)
    
    # If no pairs need processing, return empty mapping
    if not pairs_to_process:
        logger.info("No answers need cleaning (all are 3 words or less)")
        return skip_mapping
    
    # Build batch prompt
    batch_items = []
    for i, qa_pair in enumerate(pairs_to_process, 1):
        batch_items.append(f"{i}. Q: \"{qa_pair.question}\" A: \"{qa_pair.answer}\"")
    
    batch_text = "\n".join(batch_items)
    
    prompt = f"""Please clean and shorten the following answers to make them as concise as possible while preserving their meaning and ensuring they make sense with their questions.

        Rules:
        1. Remove unnecessary words like "Yes,", "I study", "I have", etc. when they don't add meaning
        2. Keep only the essential information
        3. Each answer should still make grammatical sense when read with its question
        4. If an answer is already concise, return it unchanged
        5. Remove redundant phrases but keep the core meaning

        Questions and Answers to clean:
        {batch_text}

        Provide only the cleaned answers in the same numbered format (1., 2., 3., etc.), nothing else:"""

    payload = {
        "model": PipelineConfig.LLM_MODEL,
        "max_tokens": PipelineConfig.MAX_TOKENS,
        "temperature": PipelineConfig.TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                PipelineConfig.LLM_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            response_data = response.json()
            batch_response = response_data['content'][0]['text'].strip()
            
            # Parse the numbered response
            answer_mapping = {}
            lines = batch_response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit():
                    try:
                        # Extract number and answer
                        parts = line.split('.', 1)
                        if len(parts) == 2:
                            num = int(parts[0]) - 1  # Convert to 0-based index
                            cleaned_answer = parts[1].strip()
                            
                            # Remove quotes if they were added
                            if cleaned_answer.startswith('"') and cleaned_answer.endswith('"'):
                                cleaned_answer = cleaned_answer[1:-1]
                            
                            # Map original to cleaned
                            if 0 <= num < len(pairs_to_process):
                                original_answer = pairs_to_process[num].answer
                                answer_mapping[original_answer] = cleaned_answer
                                
                                if original_answer != cleaned_answer:
                                    logger.debug(f"Cleaned: '{original_answer}' -> '{cleaned_answer}'")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing line '{line}': {e}")
                        continue
            
            # Add skipped answers to mapping
            answer_mapping.update(skip_mapping)
            
            logger.info(f"Batch processed {len(pairs_to_process)} answers, modified {len([k for k, v in answer_mapping.items() if k != v])} answers")
            return answer_mapping
            
    except Exception as e:
        logger.warning(f"Error in batch cleaning: {e}")
        # Fallback: return original answers
        fallback_mapping = {qa_pair.answer: qa_pair.answer for qa_pair in qa_pairs}
        return fallback_mapping

@task(name="process_qa_batch")
async def process_qa_batch(
    qa_pairs: List[QAPair], 
    api_key: str, 
    batch_delay: float = 1.0,
    processed_transcript_ids: Optional[Set[str]] = None
    ) -> Tuple[List[QAPair], Dict[str, str]]:
    """Process QA pairs in batches for efficiency and clean answers."""
    logger = get_run_logger()
    
    # Separate already processed pairs from new ones based on transcript_id
    already_processed = []
    new_pairs_to_process = []

    if processed_transcript_ids:
        for qa_pair in qa_pairs:
            if qa_pair.transcript_id in processed_transcript_ids:
                already_processed.append(qa_pair)
            else:
                new_pairs_to_process.append(qa_pair)
        
        logger.info(f"Found {len(already_processed)} QA pairs from {len(processed_transcript_ids)} already processed transcripts")
        logger.info(f"Will process {len(new_pairs_to_process)} new QA pairs from unprocessed transcripts")
    else:
        new_pairs_to_process = qa_pairs
        logger.info(f"No processed transcript IDs provided, will process all {len(qa_pairs)} QA pairs")
    
    if not new_pairs_to_process:
        logger.info("No new QA pairs to process")
        return already_processed, {}
    
    newly_processed_pairs = []
    all_answer_mapping = {}
    
    batch_size = 20  # Adjust based on your typical answer length and LLM token limits
    
    logger.info(f"Starting to process {len(new_pairs_to_process)} QA pairs in batches of {batch_size} for cleaning.")
    
    for i in range(0, len(new_pairs_to_process), batch_size):
        batch = new_pairs_to_process[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(new_pairs_to_process) + batch_size - 1) // batch_size
        
        logger.info(f"Processing cleaning batch {batch_num}/{total_batches} ({len(batch)} pairs)")
        
        # Clean answers in batch
        answer_mapping = await clean_answers_batch(batch, api_key)
        
        # Create processed pairs with cleaned answers
        for qa_pair in batch:
            cleaned_answer = answer_mapping.get(qa_pair.answer, qa_pair.answer)
            
            cleaned_pair = QAPair(
                question=qa_pair.question,
                answer=cleaned_answer,
                reason=qa_pair.reason,
                transcript_id=qa_pair.transcript_id
            )
            
            newly_processed_pairs.append(cleaned_pair)
        
        # Merge answer mappings
        all_answer_mapping.update(answer_mapping)
        
        # Add delay between batches to respect rate limits
        if i + batch_size < len(new_pairs_to_process):
            logger.debug(f"Cleaning batch delay: {batch_delay}s")
            await asyncio.sleep(batch_delay)
    
    
    # Combine all pairs: already processed + newly processed
    all_pairs = already_processed + newly_processed_pairs

    # Filter out unchanged answers from mapping for final output
    final_cleaning_mapping = {k: v for k, v in all_answer_mapping.items() if k != v}
    
    logger.info(f"Cleaning complete. Modified {len(final_cleaning_mapping)} new answers across {total_batches} batches.")
    logger.info(f"Returning {len(all_pairs)} total QA pairs ({len(already_processed)} existing + {len(newly_processed_pairs)} newly processed)")
    return all_pairs, final_cleaning_mapping

# NEW TASK: flatten_answers
@task(name="flatten_answers", retries=3, retry_delay_seconds=PipelineConfig.RATE_LIMIT_DELAY_LONG)
async def flatten_answers(qa_pairs: List[QAPair], api_key: str,processed_transcript_ids: Optional[Set[str]] = None) -> Tuple[List[QAPair], Dict[str, str]]:
    """ Uses an LLM to flatten semantically similar answers into canonical forms. 
    Processes unique answers in batches, identifying existing canonical forms 
    or generating new ones."""
    logger = get_run_logger()

    # Separate already processed pairs from new ones based on transcript_id
    already_processed = []
    new_pairs_to_process = []
    
    if processed_transcript_ids:
        for qa_pair in qa_pairs:
            if qa_pair.transcript_id in processed_transcript_ids:
                already_processed.append(qa_pair)
            else:
                new_pairs_to_process.append(qa_pair)
        
        logger.info(f"Found {len(already_processed)} QA pairs from already processed transcripts for flattening")
        logger.info(f"Will flatten {len(new_pairs_to_process)} new QA pairs from unprocessed transcripts")
    else:
        new_pairs_to_process = qa_pairs
        logger.info(f"No processed transcript IDs provided, will flatten all {len(qa_pairs)} QA pairs")

    if not new_pairs_to_process:
        logger.info("No new QA pairs to process for flattening")
        return already_processed, {}

    # Get all unique answers from the new QA pairs only
    unique_answers = sorted(list(set(qa_pair.answer for qa_pair in new_pairs_to_process)))


    if not unique_answers:
        logger.info("No unique answers found for flattening.")
        return already_processed + new_pairs_to_process, {}

    # This map will store the final mapping from any original answer text to its chosen canonical text
    final_answer_to_canonical_map: Dict[str, str] = {}
    # This list will store the unique canonical answer texts identified so far
    master_canonical_forms_list: List[str] = []

    # Batch processing configuration for LLM calls for grouping
    # This batch size is for the answers sent to the LLM at once for categorization
    batch_size_for_llm_grouping = 10 

    # Iterate through unique answers in batches
    for i in range(0, len(unique_answers), batch_size_for_llm_grouping):
        batch_of_answers_to_categorize = unique_answers[i:i + batch_size_for_llm_grouping]
        batch_num = (i // batch_size_for_llm_grouping) + 1
        total_batches = (len(unique_answers) + batch_size_for_llm_grouping - 1) // batch_size_for_llm_grouping

        logger.info(f"Processing unique answer batch {batch_num}/{total_batches} for flattening ({len(batch_of_answers_to_categorize)} answers in this batch)")

        # Prepare the prompt for the LLM
        # The prompt includes the *globally* established canonical forms for context
        prompt = f"""You are an expert in semantic analysis, tasked with identifying and grouping semantically equivalent answers.

            You will be given a list of `existing_canonical_forms` that have already been established from previous analysis, and a `new_answers_batch` which are answers that need to be categorized within this current batch.

            For each answer in the `new_answers_batch`, your goal is to:
            1. Determine if it is semantically identical or very similar in meaning to any of the answers in `existing_canonical_forms`.
               - If YES: Map this `new_answer` to the *exact text* of the most similar answer from `existing_canonical_forms`.
               - If NO: Create a concise, representative canonical form for this `new_answer`. This new canonical form should be short but retain the full meaning. It might be the `new_answer` itself if it's already good, or a slight rephrasing.

            Your output must be a JSON object with the following structure:
            {{
                "mapping": {{
                    "New Answer 1 from batch": "Assigned Canonical Form A",
                    "New Answer 2 from batch": "Assigned Canonical Form B",
                    ...
                }},
                "new_canonical_forms_generated_in_this_batch": [
                    "Concise Canonical Form X",
                    "Concise Canonical Form Y"
                    // List of any *new* canonical forms that were created because no existing one matched
                ]
            }}

            Rules for similarity:
            - "Semantically identical or very similar" means they convey the exact same core information and intent.
            - Minor phrasing variations, reordering of clauses (if meaning is preserved), or slight differences in level of detail are acceptable for a match.
            - If an answer introduces distinct new information, even if related, it should result in a new canonical form.
            - The chosen canonical form should be as concise as possible while being comprehensive for its group.

            Existing Canonical Forms (globally established so far):
            {json.dumps(master_canonical_forms_list, indent=2) if master_canonical_forms_list else "None yet."}

            New Answers Batch to Process:
            {json.dumps(batch_of_answers_to_categorize, indent=2)}

            Ensure your output is a valid JSON object ONLY.
            Return only the JSON array, no additional text. Make sure that the response can be parsed directly, so it starts and ends with curled brackets.

            """

        payload = {
            "model": PipelineConfig.LLM_MODEL,
            "max_tokens": PipelineConfig.MAX_TOKENS,
            "temperature": PipelineConfig.TEMPERATURE, # Keep low for consistent grouping
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            async with httpx.AsyncClient(timeout=180.0) as client: # Increased timeout for potentially longer prompts/responses
                response = await client.post(
                    PipelineConfig.LLM_API_URL,
                    headers={
                        "x-api-key": api_key,
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01"
                    },
                    json=payload
                )
                response.raise_for_status() # Raise an exception for HTTP errors

                llm_response_data = response.json()
                llm_response_text = llm_response_data['content'][0]['text'].strip()


                # Use a regex to extract the JSON string from within the markdown block
                # It looks for ```json followed by any characters (non-greedy) until ```
                json_match = re.search(r'```json\n(.*?)\n```', llm_response_text, re.DOTALL)
                if json_match:
                    json_string_to_parse = json_match.group(1).strip()
                    logger.debug("Successfully extracted JSON from markdown block.")
                else:
                    # If no markdown block is found, assume the response is pure JSON
                    # This handles cases where the LLM might sometimes not use the markdown wrapper
                    json_string_to_parse = llm_response_text
                    logger.warning("LLM response did not contain expected markdown JSON block. Attempting to parse as raw JSON.")


                try:
                    parsed_llm_output = json.loads(json_string_to_parse)
                    batch_mapping = parsed_llm_output.get("mapping", {})
                    newly_generated_canonicals = parsed_llm_output.get("new_canonical_forms_generated_in_this_batch", [])

                    # Update master canonical forms list with new ones generated in this batch
                    for new_canon in newly_generated_canonicals:
                        if new_canon not in master_canonical_forms_list:
                            master_canonical_forms_list.append(new_canon)
                            logger.debug(f"Added new master canonical form: '{new_canon}'")

                    # Apply the mapping from the LLM for this batch
                    for original_ans, assigned_canon in batch_mapping.items():
                        # Ensure the original answer was actually part of this batch (for safety)
                        if original_ans in batch_of_answers_to_categorize:
                            # If the assigned canonical form is not yet in our master list, add it.
                            if assigned_canon not in master_canonical_forms_list:
                                master_canonical_forms_list.append(assigned_canon)
                                logger.debug(f"Implicitly added new master canonical form from mapping: '{assigned_canon}'")

                            final_answer_to_canonical_map[original_ans] = assigned_canon
                            if original_ans != assigned_canon:
                                logger.debug(f"Flattened: '{original_ans}' -> '{assigned_canon}'")
                        else:
                            logger.warning(f"LLM returned mapping for unexpected answer: '{original_ans}' in batch {batch_num}. Skipping.")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response for batch {batch_num}: {e}. Response: '{llm_response_text}'")
                    # Fallback: if parsing fails, treat each answer in the current batch as its own canonical form
                    for ans in batch_of_answers_to_categorize:
                        if ans not in final_answer_to_canonical_map:
                            final_answer_to_canonical_map[ans] = ans
                            if ans not in master_canonical_forms_list:
                                master_canonical_forms_list.append(ans)
                except Exception as e:
                    logger.error(f"Error processing LLM output for batch {batch_num}: {e}. Raw response: '{llm_response_text}'")
                    # Fallback
                    for ans in batch_of_answers_to_categorize:
                        if ans not in final_answer_to_canonical_map:
                            final_answer_to_canonical_map[ans] = ans
                            if ans not in master_canonical_forms_list:
                                master_canonical_forms_list.append(ans)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during flattening batch {batch_num}: {e.response.status_code} - {e.response.text}")
            # Fallback: if API call fails, assume each answer in the batch is its own canonical
            for ans in batch_of_answers_to_categorize:
                if ans not in final_answer_to_canonical_map:
                    final_answer_to_canonical_map[ans] = ans
                    if ans not in master_canonical_forms_list:
                        master_canonical_forms_list.append(ans)
        except httpx.RequestError as e:
            logger.error(f"Request error during flattening batch {batch_num}: {e}")
            # Fallback
            for ans in batch_of_answers_to_categorize:
                if ans not in final_answer_to_canonical_map:
                    final_answer_to_canonical_map[ans] = ans
                    if ans not in master_canonical_forms_list:
                        master_canonical_forms_list.append(ans)
        except Exception as e:
            logger.error(f"Unexpected error during flattening batch {batch_num}: {e}")
            # Fallback
            for ans in batch_of_answers_to_categorize:
                if ans not in final_answer_to_canonical_map:
                    final_answer_to_canonical_map[ans] = ans
                    if ans not in master_canonical_forms_list:
                        master_canonical_forms_list.append(ans)

        # Add delay between batches to respect rate limits
        if i + batch_size_for_llm_grouping < len(unique_answers):
            logger.debug(f"Flattening batch delay: {PipelineConfig.RATE_LIMIT_DELAY_LONG}s")
            await asyncio.sleep(PipelineConfig.RATE_LIMIT_DELAY_LONG)

    # Apply flattening mapping to new pairs only
    newly_flattened_pairs = []
    for qa_pair in new_pairs_to_process:
        flattened_answer = final_answer_to_canonical_map.get(qa_pair.answer, qa_pair.answer)
        flattened_pair = QAPair(
            question=qa_pair.question,
            answer=flattened_answer,
            reason=qa_pair.reason,
            transcript_id=qa_pair.transcript_id
        )
        newly_flattened_pairs.append(flattened_pair)
    
    # Combine all pairs: already processed + newly flattened
    all_pairs = already_processed + newly_flattened_pairs
    
    # Only return mappings for newly processed answers
    final_flattening_mapping = {k: v for k, v in final_answer_to_canonical_map.items() if k != v}
    
    logger.info(f"Flattening complete. Modified {len(final_flattening_mapping)} new answers.")
    logger.info(f"Returning {len(all_pairs)} total QA pairs ({len(already_processed)} existing + {len(newly_flattened_pairs)} newly flattened)")
    return all_pairs, final_flattening_mapping

@task(name="flatten_question_batch", retries=3, retry_delay_seconds=PipelineConfig.RATE_LIMIT_DELAY_LONG*10)
async def flatten_question_batch(
    unique_questions_batch: List[str],
    current_master_canonical_forms: List[str],
    api_key: str,
    batch_num: int,
    total_batches: int
) -> Tuple[Dict[str, str], List[str]]:
    logger = get_run_logger()

    logger.info(f"Processing unique question batch {batch_num}/{total_batches} for flattening ({len(unique_questions_batch)} questions in this batch)")

    prompt = f"""You are an expert in semantic analysis, tasked with identifying and grouping semantically equivalent questions. Your primary goal is to create canonical forms that accurately reflect the SPECIFIC INTENT AND TYPE OF INFORMATION SOUGHT by each question.

    You will be given a list of `existing_canonical_question_forms` that have already been established from previous analysis, and a `new_questions_batch` which are questions that need to be categorized within this current batch.

    For each question in the `new_questions_batch`, your goal is to:
    1. Determine if it is semantically identical or very similar in meaning to any of the questions in `existing_canonical_question_forms`.
        - If YES: Map this `new_question` to the *exact text* of the most similar question from `existing_canonical_question_forms`.
        - If NO: Create a concise, representative canonical form for this `new_question`. This new canonical form should be short but retain the full meaning AND SPECIFIC INTENT. It might be the `new_question` itself if it's already good, or a slight rephrasing.

    Your output must be a JSON object with the following structure:
    {{
        "mapping": {{
            "New question 1 from batch": "Assigned Canonical Form A",
            "New question 2 from batch": "Assigned Canonical Form B",
            ...
        }},
        "new_canonical_forms_generated_in_this_batch": [
            "Concise Canonical Form X",
            "Concise Canonical Form Y"
            // List of any *new* canonical forms that were created because no existing one matched
        ]
    }}

    Rules for similarity:
    - "Semantically identical or very similar" means they convey the exact same core information and primary intent.
    - Questions should *only* be grouped if they are asking for the *same type of information* about the *same specific subject or action*.
    - Minor phrasing variations, reordering of clauses (if meaning is preserved), or slight differences in level of detail are acceptable for a match, BUT ONLY IF THE UNDERLYING REQUESTED INFORMATION TYPE AND SPECIFIC TOPIC REMAIN UNCHANGED.
    - If a question is asking about a different specific action, event, entity, or type of feedback (e.g., asking about *attendance at events* vs. asking for *general feedback to faculty*), it must result in a new canonical form, even if a common institution is mentioned.
    - The chosen canonical form should be as concise as possible while being comprehensive for its group, and must capture the specific intent and subject.

    Existing Canonical Forms (globally established so far):
    {json.dumps(current_master_canonical_forms, indent=2) if current_master_canonical_forms else "None yet."}

    New Questions Batch to Process:
    {json.dumps(unique_questions_batch, indent=2)}

    Ensure your output is a valid JSON object ONLY. Do not include any other text.
    """

    payload = {
        "model": PipelineConfig.LLM_MODEL,
        "max_tokens": PipelineConfig.MAX_TOKENS,
        "temperature": PipelineConfig.TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}]
    }

    # Initialize local batch results
    batch_question_to_canonical_map: Dict[str, str] = {}
    newly_generated_canonicals_in_this_batch: List[str] = []

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                PipelineConfig.LLM_API_URL,
                headers={
                    "x-api-key": api_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json=payload
            )
            response.raise_for_status()

            llm_response_data = response.json()
            llm_response_text = llm_response_data['content'][0]['text'].strip()

            json_match = re.search(r'```json\n(.*?)\n```', llm_response_text, re.DOTALL)
            if json_match:
                json_string_to_parse = json_match.group(1).strip()
                logger.debug("Successfully extracted JSON from markdown block.")
            else:
                json_string_to_parse = llm_response_text
                logger.warning("LLM response did not contain expected markdown JSON block. Attempting to parse as raw JSON.")

            parsed_llm_output = json.loads(json_string_to_parse)
            batch_mapping = parsed_llm_output.get("mapping", {})
            newly_generated_canonicals_from_llm = parsed_llm_output.get("new_canonical_forms_generated_in_this_batch", [])

            # Add newly generated canonicals from LLM to our local list for this batch
            newly_generated_canonicals_in_this_batch.extend(newly_generated_canonicals_from_llm)

            # Apply the mapping from the LLM for this batch
            for original_q, assigned_canon in batch_mapping.items():
                if original_q in unique_questions_batch:
                    batch_question_to_canonical_map[original_q] = assigned_canon
                    if original_q != assigned_canon:
                        logger.debug(f"Flattened: '{original_q}' -> '{assigned_canon}'")
                else:
                    logger.warning(f"LLM returned mapping for unexpected question: '{original_q}' in batch {batch_num}. Skipping.")

            # If execution reaches here, it means the LLM call and parsing were successful for this batch
            return batch_question_to_canonical_map, newly_generated_canonicals_in_this_batch

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response for batch {batch_num}: {e}. Response: '{llm_response_text}'")
        # **Crucial change:** Re-raise the exception to trigger Prefect's retry mechanism
        raise e
        # Alternatively, return Prefect's Failed state:
        # return Failed(message=f"Failed to parse LLM JSON for batch {batch_num}: {e}", data=(batch_question_to_canonical_map, newly_generated_canonicals_in_this_batch))

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during flattening batch {batch_num}: {e.response.status_code} - {e.response.text}")
        # Re-raise to trigger retries
        raise e

    except httpx.RequestError as e:
        logger.error(f"Request error during flattening batch {batch_num}: {e}")
        # Re-raise to trigger retries
        raise e

    except Exception as e:
        logger.error(f"Unexpected error during flattening batch {batch_num}: {e}")
        # Re-raise to trigger retries
        raise e

@task(name="flatten_questions", retries=3, retry_delay_seconds=PipelineConfig.RATE_LIMIT_DELAY_LONG)
async def flatten_questions(
    qa_pairs: List[QAPair],
    api_key: str,
    processed_transcript_ids: Optional[Set[str]] = None
) -> Tuple[List[QAPair], Dict[str, str]]:
    """
    Uses an LLM to flatten semantically similar questions into canonical forms.
    Processes unique questions in batches, identifying existing canonical forms
    or generating new ones.
    """
    logger = get_run_logger()

    already_processed = []
    new_pairs_to_process = []

    if processed_transcript_ids:
        for qa_pair in qa_pairs:
            if qa_pair.transcript_id in processed_transcript_ids:
                already_processed.append(qa_pair)
            else:
                new_pairs_to_process.append(qa_pair)

        logger.info(f"Found {len(already_processed)} QA pairs from already processed transcripts for flattening")
        logger.info(f"Will flatten {len(new_pairs_to_process)} new QA pairs from unprocessed transcripts")
    else:
        new_pairs_to_process = qa_pairs
        logger.info(f"No processed transcript IDs provided, will flatten all {len(qa_pairs)} QA pairs")

    if not new_pairs_to_process:
        logger.info("No new QA pairs to process for flattening")
        return already_processed, {}

    unique_questions_strings = sorted(list(set(qa_pair.question for qa_pair in new_pairs_to_process)))

    if not unique_questions_strings:
        logger.info("No unique questions found for flattening.")
        return already_processed + new_pairs_to_process, {}

    final_question_to_canonical_map: Dict[str, str] = {}
    master_canonical_forms_list: List[str] = [] # This will be cumulatively updated

    batch_size_for_llm_grouping = 7

    total_unique_questions = len(unique_questions_strings)
    total_batches = (total_unique_questions + batch_size_for_llm_grouping - 1) // batch_size_for_llm_grouping

    # --- KEY CHANGE: Sequential processing and immediate merging ---
    for i in range(0, total_unique_questions, batch_size_for_llm_grouping):
        current_batch_questions = unique_questions_strings[i:i + batch_size_for_llm_grouping]
        current_batch_num = (i // batch_size_for_llm_grouping) + 1

        # AWAIT the task directly in the loop. This means each batch runs sequentially.
        # The returned values immediately update the master lists for the *next* iteration.
        batch_mapping, newly_generated_canonicals_from_batch = await flatten_question_batch(
            unique_questions_batch=current_batch_questions,
            current_master_canonical_forms=list(master_canonical_forms_list), # Still pass a copy to the task!
            api_key=api_key,
            batch_num=current_batch_num,
            total_batches=total_batches
        )

        # Merge the results from the *just completed* batch immediately
        final_question_to_canonical_map.update(batch_mapping)

        for new_canon in newly_generated_canonicals_from_batch:
            if new_canon not in master_canonical_forms_list:
                master_canonical_forms_list.append(new_canon)
                # logger.debug(f"Added new master canonical form: '{new_canon}' after processing batch.")
        
        # Also ensure any assigned canonicals from the mapping are in the master list
        for original_q, assigned_canon in batch_mapping.items():
            if assigned_canon not in master_canonical_forms_list:
                master_canonical_forms_list.append(assigned_canon)
                # logger.debug(f"Implicitly added new master canonical form from mapping: '{assigned_canon}' after processing batch.")

        # Add delay between batches to respect rate limits if needed,
        # otherwise Prefect's retries/flow settings should handle it
        # You removed this from the batch task, which is fine, but if you need an explicit delay
        # between sequential calls, put it here.
        # For Claude's 529 errors, this might be beneficial.
        if i + batch_size_for_llm_grouping < total_unique_questions:
             logger.debug(f"Delaying {PipelineConfig.RATE_LIMIT_DELAY_LONG}s before next batch.")
             await asyncio.sleep(PipelineConfig.RATE_LIMIT_DELAY_LONG)

    # --- Rest of the function remains largely the same ---
    newly_flattened_pairs = []
    for qa_pair in new_pairs_to_process:
        flattened_question = final_question_to_canonical_map.get(qa_pair.question, qa_pair.question)
        flattened_pair = QAPair(
            question=flattened_question,
            answer=qa_pair.answer,
            reason=qa_pair.reason,
            transcript_id=qa_pair.transcript_id
        )
        newly_flattened_pairs.append(flattened_pair)

    all_pairs = already_processed + newly_flattened_pairs
    final_flattening_mapping = {k: v for k, v in final_question_to_canonical_map.items() if k != v}

    logger.info(f"Flattening complete. Modified {len(final_flattening_mapping)} new questions.")
    logger.info(f"Returning {len(all_pairs)} total QA pairs ({len(already_processed)} existing + {len(newly_flattened_pairs)} newly flattened)")
    return all_pairs, final_flattening_mapping

@task(name="save_processed_csv")
def save_data_to_csv(qa_pairs: List[QAPair], output_path: str) -> str:
    """Save processed QA pairs to CSV file."""
    logger = get_run_logger()
    
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['question', 'answer', 'reason', 'transcript_id', 'content_category']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()
            for qa_pair in qa_pairs:
                writer.writerow({
                    'question': qa_pair.question,
                    'answer': qa_pair.answer,
                    'reason': qa_pair.reason,
                    'transcript_id': qa_pair.transcript_id,
                    'content_category':qa_pair.content_category
                })
        
        logger.info(f"Processed data saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving CSV file: {e}")
        raise

@task(name='add-categories', retries=3, retry_delay_seconds=PipelineConfig.RATE_LIMIT_DELAY_LONG*10)
async def add_question_categories(
    qa_pairs: List[QAPair],
    api_key: str,
    processed_transcript_ids: Optional[Set[str]] = None,
    merge_with_existing: bool = False
    ) -> Tuple[List[QAPair], Dict[str, str]]:
    
    logger = get_run_logger()
    
    # Filter qa_pairs based on processed_transcript_ids
    already_processed = []
    new_pairs_to_process = []
    
    if processed_transcript_ids:
        for qa_pair in qa_pairs:
            if qa_pair.transcript_id in processed_transcript_ids:
                already_processed.append(qa_pair)
            else:
                new_pairs_to_process.append(qa_pair)
    else:
        new_pairs_to_process = qa_pairs
    
    # Get unique questions from pairs to process
    unique_questions_strings = sorted(list(set(qa_pair.question for qa_pair in new_pairs_to_process)))
    total_unique_questions = len(unique_questions_strings)
    context_information = '''Transcripts of voice based conversations with a Feedback Bot and students, on the topic of their studies with the goal of improving teaching'''
    
    logger.info(f"Identifying categories for {total_unique_questions} questions.")
    
    # Load existing categories if merge_with_existing is True
    existing_question_to_category = {}
    categories_file_path = "existing_categories.json"
    categories_list_file_path = "categories_list.txt"
    
    if merge_with_existing:
        try:
            if os.path.exists(categories_file_path):
                with open(categories_file_path, 'r', encoding='utf-8') as f:
                    existing_question_to_category = json.load(f)
                logger.info(f"Loaded {len(existing_question_to_category)} existing question-category mappings.")
        except Exception as e:
            logger.warning(f"Could not load existing categories: {e}")
            existing_question_to_category = {}
    
    # Filter out questions that already have categories (if merging)
    questions_needing_categorization = []
    if merge_with_existing:
        for question in unique_questions_strings:
            if question not in existing_question_to_category:
                questions_needing_categorization.append(question)
    else:
        questions_needing_categorization = unique_questions_strings
    
    # Initialize question to category mapping with existing data
    question_to_category_map = existing_question_to_category.copy()
    
    # Only call LLM if there are new questions to categorize
    if questions_needing_categorization:
        prompt = f"""
        Take these questions and group them into categories. 
        Context: {context_information}
        New Questions Batch to Process:
        {json.dumps(questions_needing_categorization, indent=2)}
        Return a JSON mapping where each question maps to its category name.
        Format: {{"question": "category_name"}}
        Ensure your output is a valid JSON object ONLY. Do not include any other text.
        """
        
        payload = {
            "model": PipelineConfig.LLM_MODEL,
            "max_tokens": PipelineConfig.MAX_TOKENS,
            "temperature": PipelineConfig.TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    PipelineConfig.LLM_API_URL,
                    headers={
                        "x-api-key": api_key,
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01"
                    },
                    json=payload
                )
                response.raise_for_status()
                llm_response_data = response.json()
                llm_response_text = llm_response_data['content'][0]['text'].strip()
                
                # Extract JSON from response
                json_match = re.search(r'```json\n(.*?)\n```', llm_response_text, re.DOTALL)
                if json_match:
                    json_string_to_parse = json_match.group(1).strip()
                    logger.debug("Successfully extracted JSON from markdown block.")
                else:
                    json_string_to_parse = llm_response_text
                    logger.warning("LLM response did not contain expected markdown JSON block. Attempting to parse as raw JSON.")
                
                # Parse LLM response
                parsed_llm_output = json.loads(json_string_to_parse)
                
                # Validate that all questions got categories
                for question in questions_needing_categorization:
                    if question in parsed_llm_output:
                        question_to_category_map[question] = parsed_llm_output[question]
                    else:
                        logger.warning(f"Question not found in LLM response, assigning default category: {question}")
                        question_to_category_map[question] = "Uncategorized"
                
                logger.info(f"Successfully categorized {len(questions_needing_categorization)} new questions.")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Assign default category to all questions needing categorization
            for question in questions_needing_categorization:
                question_to_category_map[question] = "Uncategorized"
        except Exception as e:
            logger.error(f"Unexpected error during categorization: {e}")
            # Assign default category to all questions needing categorization
            for question in questions_needing_categorization:
                question_to_category_map[question] = "Uncategorized"
            raise e
    
    # Save updated categories to files
    try:
        # Save question-to-category mapping
        with open(categories_file_path, 'w', encoding='utf-8') as f:
            json.dump(question_to_category_map, f, indent=2, ensure_ascii=False)
        
        # Save unique categories list for easy manual manipulation
        unique_categories = sorted(list(set(question_to_category_map.values())))
        with open(categories_list_file_path, 'w', encoding='utf-8') as f:
            for category in unique_categories:
                f.write(f"{category}\n")
        
        logger.info(f"Saved {len(question_to_category_map)} question-category mappings to {categories_file_path}")
        logger.info(f"Saved {len(unique_categories)} unique categories to {categories_list_file_path}")
    except Exception as e:
        logger.error(f"Failed to save categories to file: {e}")
    
    # Update qa_pairs with categories
    updated_qa_pairs = []
    
    # Add already processed pairs (unchanged)
    updated_qa_pairs.extend(already_processed)
    
    # Add newly processed pairs with categories
    for qa_pair in new_pairs_to_process:
        if qa_pair.question in question_to_category_map:
            qa_pair.content_category = question_to_category_map[qa_pair.question]
        else:
            logger.warning(f"No category found for question, assigning default: {qa_pair.question}")
            qa_pair.content_category = "Uncategorized"
        updated_qa_pairs.append(qa_pair)
    
    logger.info(f"Updated {len(new_pairs_to_process)} qa_pairs with categories.")
    
    return updated_qa_pairs, question_to_category_map

@task(name="save_answer_mapping")
def save_mapping_to_json(mapping: Dict[str, str], output_path: str, map_name: str = "Answer Mapping") -> str:
    """Save the given mapping to a JSON file."""
    logger = get_run_logger()
    
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(mapping, file, indent=2, ensure_ascii=False)
        
        logger.info(f"{map_name} saved to: {output_path}")
        logger.info(f"Total entries in {map_name}: {len(mapping)}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving {map_name} file: {e}")
        raise


@task(name="validate_api_key")
async def validate_api_key(api_key: Optional[str] = None) -> str:
    """Validate and retrieve API key."""
    logger = get_run_logger()
    
    if api_key:
        return api_key
    
    # Try to get from Prefect Secret block
    try:
        api_key_block = await Secret.load("anthropic-api-key")
        api_key = api_key_block.get()
        
        if not api_key:
            raise ValueError("API key is empty or None")
        
        logger.info("API key loaded from Prefect secret block")
        return api_key
        
    except Exception as e:
        logger.warning(f"Failed to load API key from secret block: {str(e)}")
        
        # Try environment variable
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            logger.info("API key loaded from environment variable")
            return api_key
        
        raise ValueError("No API key found. Please provide via parameter, Prefect secret block, or ANTHROPIC_API_KEY environment variable")

@flow(name="process_and_flatten_answers", log_prints=True)
async def clean_dataset_answers_flow(
    input_csv_path: str,
    output_csv_path: str = "", # Default to empty to generate dynamic name
    cleaning_mapping_json_path: str = "", # Default to empty to generate dynamic name
    flattening_answer_mapping_json_path: str = "", # Default to empty to generate dynamic name
    flattening_question_mapping_json_path: str = "", # Default to empty to generate dynamic name
    api_key: Optional[str] = None,
    batch_delay: float = 1.0 # Delay for general API calls
) -> Dict[str, str]:
    """
    Prefect flow to clean and summarize dataset answers using Claude API, and then flatten semantically similar answers.
    
    Args:
        input_csv_path: Path to input CSV file.
        output_csv_path: Path for the final cleaned and flattened CSV file.
                         If empty, a dynamic name based on timestamp will be used.
        cleaning_mapping_json_path: Path for the JSON file storing original-to-cleaned answer mappings.
                                    If empty, a dynamic name based on timestamp will be used.
        flattening_answer_mapping_json_path: Path for the JSON file storing cleaned-to-flattened answer mappings.
                                      If empty, a dynamic name based on timestamp will be used.
        api_key: Anthropic API key (if None, tries Prefect secret block then ANTHROPIC_API_KEY env var).
        batch_delay: Delay between batches in seconds (default: 1.0).
    
    Returns:
        Dictionary containing paths to generated files and summary statistics.
    """
    
    logger = get_run_logger()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not output_csv_path:
        output_csv_path = f"output_data/qa_pairs_cleaned_and_flattened_{current_time}.csv"
    if not cleaning_mapping_json_path:
        cleaning_mapping_json_path = f"output_data/answer_cleaning_map_{current_time}.json"
    if not flattening_answer_mapping_json_path:
        flattening_answer_mapping_json_path = f"output_data/answer_flattening_map_{current_time}.json"
    if not flattening_question_mapping_json_path:
        flattening_question_mapping_json_path = f"output_data/question_flattening_map_{current_time}.json"

    logger.info(f"Starting flow with input CSV: {input_csv_path}")
    logger.info(f"Output CSV will be: {output_csv_path}")

    # Validate API key
    validated_api_key = await validate_api_key(api_key)
    
    # Load data from CSV
    qa_pairs = import_qa_pairs_from_csv(input_csv_path)
    logger.info(f"Loaded {len(qa_pairs)} total QA pairs from input CSV")

    # Load existing data
    processed_ids = load_processed_transcript_ids("already_processed_transcripts.txt")
    logger.info(f"Found {len(processed_ids)} already processed transcript IDs")
    
    # 1. Clean answers (shorten/summarize individually)
    cleaned_qa_pairs, cleaning_answer_mapping = await process_qa_batch(
        qa_pairs, 
        validated_api_key, 
        batch_delay, 
        processed_transcript_ids=processed_ids
    )
    save_mapping_to_json(cleaning_answer_mapping, cleaning_mapping_json_path, map_name="Cleaning Answer Mapping")

    # 2. Flatten answers (group semantically similar ones from the 'cleaned' set)
    flattened_qa_pairs, flattening_answer_mapping = await flatten_answers(
        cleaned_qa_pairs, 
        validated_api_key, 
        processed_transcript_ids=processed_ids, # Pass the *cleaned* QA pairs to flatten
    )

    save_mapping_to_json(flattening_answer_mapping, flattening_answer_mapping_json_path, map_name="Flattening Answer Mapping")

    # 3. Flatten questions (group semantically similar ones from the 'cleaned' set)
    flattened_qa_pairs, flattening_question_mapping = await flatten_questions(
        flattened_qa_pairs, #cleaned_qa_pairs, 
        validated_api_key, 
        processed_transcript_ids= processed_ids, # Pass the *cleaned* QA pairs to flatten
    )

    save_mapping_to_json(flattening_question_mapping, flattening_question_mapping_json_path, map_name="Flattening Question Mapping")

    categorized_qa_pairs, question_to_category_map = await add_question_categories(cleaned_qa_pairs,validated_api_key)

    # save_mapping_to_json(question_to_category_map, question_to_category_map_json_path, map_name="Category Mapping")

    # Save the final (cleaned and flattened) QA pairs to a new CSV file
    final_csv_output_path = save_data_to_csv(categorized_qa_pairs, output_csv_path)

    # Update processed transcript IDs file with newly processed ones
    newly_processed_ids = set(qa.transcript_id for qa in flattened_qa_pairs if qa.transcript_id not in processed_ids)
    all_processed_ids = processed_ids.union(newly_processed_ids)
    save_processed_transcript_ids(all_processed_ids, "already_processed_transcripts.txt")
    

    return {
        "final_processed_csv_path": final_csv_output_path,
        "cleaning_mapping_json_path": cleaning_mapping_json_path,
        "flattening_answer_mapping_json_path": flattening_answer_mapping_json_path,
        "total_input_qa_pairs": len(qa_pairs),
        "total_already_processed_transcript_ids": len(processed_ids),
        "total_newly_processed_transcript_ids": len(newly_processed_ids),
        "total_answers_cleaned_modified": len(cleaning_answer_mapping),
        "total_answers_flattened_modified": len(flattening_answer_mapping),
        "total_final_qa_pairs": len(flattened_qa_pairs),
        "total_final_unique_answers_after_flattening": len(set(qp.answer for qp in flattened_qa_pairs))
    }


# Usage
if __name__ == "__main__":
    print("Waiting 3 seconds to avoid database locks...")
    time.sleep(3)

    # To run with your own CSV file:
    # Make sure to update the input CSV path to an existing file you want to process.
    # Output file paths will be dynamically generated if not specified.
    results = asyncio.run(clean_dataset_answers_flow(
        input_csv_path="output_data/qa_pairs_cleaned_and_flattened_20250620_163533.csv" # !!! IMPORTANT: Change this to your actual input CSV file path !!!
        # output_csv_path="qa_pairs_cleaned_and_flattened_manual_name.csv", # Optional: specify output names
        # cleaning_mapping_json_path="answer_cleaning_map_manual_name.json",
        # flattening_answer_mapping_json_path="answer_flattening_map_manual_name.json"
    ))
    
    print("Flow completed successfully!")
    print(f"Results: {results}")