import asyncio
import json
import pandas as pd
# import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime#, timedelta
import httpx
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.blocks.system import Secret
# import networkx as nx
from collections import Counter
import re
import csv

# Configuration
class PipelineConfig:
    LLM_API_URL = "https://api.anthropic.com/v1/messages"
    LLM_MODEL = "claude-opus-4-20250514"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.1
    RATE_LIMIT_DELAY = 1  # seconds between API calls
    RATE_LIMIT_DELAY_LONG = 5  # seconds between API calls (used for transcript processing)
    BATCH_SIZE = 10

# Data Models
class TranscriptData:
    def __init__(self, transcript_id: str, content: str, metadata: Dict):
        self.transcript_id = transcript_id
        self.content = content
        self.metadata = metadata

class QAPair:
    def __init__(self, question: str, answer: str, reason: str, transcript_id: str):
        self.question = question
        self.answer = answer
        self.reason = reason
        self.transcript_id = transcript_id


class FlatteningTransformation:
    def __init__(self, 
                 original_qa: QAPair, 
                 flattened_qa: QAPair, 
                 transformation_type: str,
                 transformation_reason: str = "",
                 group_id: str = None):
        self.original_qa = original_qa
        self.flattened_qa = flattened_qa
        self.transformation_type = transformation_type  # "unchanged", "question_standardized", "answer_standardized", "both_standardized"
        self.transformation_reason = transformation_reason
        self.group_id = group_id  # For tracking which pairs were grouped together

class FlatteningResults:
    def __init__(self):
        self.transformations: List[FlatteningTransformation] = []
        self.question_mappings: Dict[str, str] = {}  # original -> standardized
        self.answer_mappings: Dict[str, str] = {}  # original -> standardized
        self.reason_mappings: Dict[str, str] = {}  # original -> standardized
        self.statistics = {
            'total_original': 0,
            'total_flattened': 0,
            'questions_standardized': 0,
            'answers_standardized': 0,
            'reasons_standardized': 0,
            'pairs_removed': 0,
            'groups_created': 0
        }
    
    def add_transformation(self, transformation: FlatteningTransformation):
        self.transformations.append(transformation)
    
    def get_flattened_pairs(self) -> List[QAPair]:
        return [t.flattened_qa for t in self.transformations if t.flattened_qa is not None]


class AnalysisResults:
    def __init__(self, stats: Dict, connections: Dict, insights: List[str]):
        self.stats = stats
        self.connections = connections
        self.insights = insights

# Step 1: Data Ingestion and Preprocessing
@task(name="load_transcripts", retries=2)
def load_transcripts(file_path: str) -> List[TranscriptData]:
    """Load and validate transcript data from file"""
    logger = get_run_logger()
    logger.info(f"Loading transcripts from {file_path}")
    
    try:
        # Assuming CSV format with columns: id, content, metadata
        df = pd.read_csv(file_path)
        transcripts = []
        
        for _, row in df.iterrows():
            metadata = json.loads(row.get('metadata', '{}'))
            transcript = TranscriptData(
                transcript_id=str(row['id']),
                content=str(row['content']),
                metadata=metadata
            )
            transcripts.append(transcript)
        
        logger.info(f"Successfully loaded {len(transcripts)} transcripts")
        return transcripts
    
    except Exception as e:
        logger.error(f"Failed to load transcripts: {str(e)}")
        raise

@task(name="clean_transcripts", retries=1)
def clean_transcripts(transcripts: List[TranscriptData]) -> List[TranscriptData]:
    """Clean and preprocess transcript content"""
    logger = get_run_logger()
    logger.info("Cleaning transcript data")
    
    cleaned_transcripts = []
    for transcript in transcripts:
        # Basic cleaning
        cleaned_content = transcript.content.strip()
        
        # Remove excessive whitespace
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        
        # Remove special characters that might interfere with LLM processing
        cleaned_content = re.sub(r'[^\w\s\.\?\!\,\:\;\-\(\)]', '', cleaned_content)
        
        # Skip if content is too short or too long
        if len(cleaned_content) < 50 or len(cleaned_content) > 50000:
            logger.warning(f"Skipping transcript {transcript.transcript_id} - invalid length")
            continue
        
        transcript.content = cleaned_content
        cleaned_transcripts.append(transcript)
    
    logger.info(f"Cleaned {len(cleaned_transcripts)} transcripts")
    return cleaned_transcripts

# Step 2: LLM Processing for Q&A Extraction
@task(name="extract_qa_pairs", retries=3, retry_delay_seconds=60)
async def extract_qa_pairs_from_transcript(transcript: TranscriptData) -> List[QAPair]:
    """Extract question-answer pairs from a single transcript using LLM"""
    logger = get_run_logger()
    content = None  # Initialize content variable
    
    # Get API key from Prefect Secret block
    try:
        api_key_block = await Secret.load("anthropic-api-key")
        api_key = api_key_block.get()
        
        if not api_key:
            raise ValueError("API key is empty or None")
        
    except Exception as e:
        logger.error(f"Failed to load API key from secret block: {str(e)}")
        raise

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    # Debug the headers (without exposing the full key)
    logger.info(f"Headers prepared with API key: {bool(headers.get('x-api-key'))}")

    prompt = f"""
    Please analyze the following feedback conversation transcript and extract all question-answer pairs along with the reasoning behind each answer.

    Format your response as a JSON array with objects containing:
    - "question": The exact question asked
    - "answer": The response given
    - "reason": The reasoning or justification provided for the answer

    In case a question has multiple answers, and in case multiple reasons are given for an answer, create multiple question, answer, reason pairs for the same question.

    Translate the answers, questions and reasons into English, if they are in another language.
    Transcript:
    {transcript.content}

    Return only the JSON array, no additional text. Make sure that the response can be parsed directly, so it starts and ends with curled backets.
    """
    # TBD improve translation
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
            
            result = response.json()
            content = result["content"][0]["text"]

            # Parse JSON response
            try:
                qa_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Raw content: {content}")
                return []  # Return empty list instead of crashing
            
            qa_pairs = []
            for item in qa_data:
                qa_pair = QAPair(
                    question=item["question"],
                    answer=item["answer"],
                    reason=item["reason"],
                    transcript_id=transcript.transcript_id
                )
                qa_pairs.append(qa_pair)
            
            logger.info(f"Extracted {len(qa_pairs)} Q&A pairs from transcript {transcript.transcript_id}")
            
            # Rate limiting
            await asyncio.sleep(PipelineConfig.RATE_LIMIT_DELAY)
            
            return qa_pairs
    
    except Exception as e:
        error_msg = f"Failed to extract Q&A pairs from transcript {transcript.transcript_id}: {str(e)}"
        if content is not None:
            error_msg += f", Raw content: {content[:500]}..."  # Limit content length in error
        logger.error(error_msg)
        raise

@task(name="batch_extract_qa", retries=2)
async def batch_extract_qa_pairs(transcripts: List[TranscriptData]) -> List[QAPair]:
    """Process multiple transcripts with controlled concurrency"""
    logger = get_run_logger()
    logger.info(f"Starting batch extraction for {len(transcripts)} transcripts")
    
    all_qa_pairs = []
    
    # Process in batches to manage rate limits and memory
    for i in range(0, len(transcripts), PipelineConfig.BATCH_SIZE):
        batch = transcripts[i:i + PipelineConfig.BATCH_SIZE]
        logger.info(f"Processing batch {i//PipelineConfig.BATCH_SIZE + 1}")
        
        # Process batch sequentially to respect rate limits
        batch_results = []
        for transcript in batch:
            try:
                qa_pairs = await extract_qa_pairs_from_transcript(transcript)
                batch_results.extend(qa_pairs)
            except Exception as e:
                logger.error(f"Failed to process transcript {transcript.transcript_id}: {str(e)}")
                continue
        
        all_qa_pairs.extend(batch_results)
        
        # Pause between batches
        if i + PipelineConfig.BATCH_SIZE < len(transcripts):
            await asyncio.sleep(PipelineConfig.RATE_LIMIT_DELAY_LONG)
    
    logger.info(f"Extracted total of {len(all_qa_pairs)} Q&A pairs")
    return all_qa_pairs



# Step 2.5: clean up and flatten the QA pairs, so that the same answers have the same wording (allow easier statistical analysis)
# needs checks

@task(name="flatten_qa_pairs_with_tracking", retries=3, retry_delay_seconds=60)
async def flatten_qa_pairs_with_tracking(qa_pairs: List[QAPair]) -> FlatteningResults:
    """Flatten question-answer pairs while tracking all transformations"""
    logger = get_run_logger()
    logger.info(f"Starting tracked flattening process for {len(qa_pairs)} Q&A pairs")
    
    results = FlatteningResults()
    results.statistics['total_original'] = len(qa_pairs)
    
    if not qa_pairs:
        logger.warning("No Q&A pairs to flatten")
        return results
    
    # Get API key
    try:
        api_key_block = await Secret.load("anthropic-api-key")
        api_key = api_key_block.get()
        if not api_key:
            raise ValueError("API key is empty or None")
    except Exception as e:
        logger.error(f"Failed to load API key from secret block: {str(e)}")
        raise

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    try:
        # Phase 1: Flatten questions and track mappings
        logger.info("Phase 1: Flattening questions globally with tracking")
        question_mapping = await _flatten_questions_with_tracking(qa_pairs, headers, logger, results)
        
        # Phase 2: Flatten answers globally and track mappings
        logger.info("Phase 2: Flattening answers globally with tracking")
        answer_mapping = await _flatten_answers_with_tracking(qa_pairs, headers, logger, results)
        
        # Phase 3: Flatten reasons globally and track mappings
        logger.info("Phase 3: Flattening reasons globally with tracking")
        reason_mapping = await _flatten_reasons_with_tracking(qa_pairs, headers, logger, results)
        
        # Phase 4: Apply all mappings and create final transformations
        logger.info("Phase 4: Applying all mappings and creating transformations")
        await _apply_mappings_with_tracking(qa_pairs, question_mapping, answer_mapping, reason_mapping, results)
        
        # Update final statistics
        results.statistics['total_flattened'] = len(results.get_flattened_pairs())
        results.statistics['questions_standardized'] = sum(1 for t in results.transformations 
                                                         if 'question' in t.transformation_type)
        results.statistics['answers_standardized'] = sum(1 for t in results.transformations 
                                                       if 'answer' in t.transformation_type)
        results.statistics['reasons_standardized'] = sum(1 for t in results.transformations 
                                                       if 'reason' in t.transformation_type)
        
        logger.info(f"Tracked flattening completed. {results.statistics}")
        return results
        
    except Exception as e:
        logger.error(f"Tracked flattening process failed: {str(e)}")
        # Create fallback transformations (unchanged)
        for qa in qa_pairs:
            results.add_transformation(FlatteningTransformation(
                original_qa=qa,
                flattened_qa=qa,
                transformation_type="unchanged",
                transformation_reason="Fallback due to processing error"
            ))
        return results



async def _flatten_questions_with_tracking(qa_pairs: List[QAPair], headers: Dict, logger, results: FlatteningResults) -> Dict[str, str]:
    """Phase 1 with tracking: Create question mappings and record transformations"""
    
    unique_questions = list(set(qa.question for qa in qa_pairs))
    logger.info(f"Found {len(unique_questions)} unique questions to analyze")
    
    if len(unique_questions) <= 1:
        question_mapping = {q: q for q in unique_questions}
        results.question_mappings = question_mapping
        return question_mapping
    
    # Process questions in batches
    batch_size = 20 #TBD - use pipeline class value
    question_mapping = {}
    
    for i in range(0, len(unique_questions), batch_size):
        batch = unique_questions[i:i + batch_size]
        logger.info(f"Processing question batch {i//batch_size + 1}/{(len(unique_questions) + batch_size - 1)//batch_size}")
        
        try:
            batch_mapping = await _flatten_question_batch(batch, headers, logger)
            question_mapping.update(batch_mapping)
            
            if i + batch_size < len(unique_questions):
                await asyncio.sleep(PipelineConfig.RATE_LIMIT_DELAY)
                
        except Exception as e:
            logger.error(f"Failed to process question batch {i//batch_size + 1}: {str(e)}")
            for question in batch:
                question_mapping[question] = question
            continue
    
    # Store question mappings in results
    results.question_mappings = question_mapping
    
    # Count question standardizations
    questions_changed = sum(1 for orig, flat in question_mapping.items() if orig != flat)
    unique_flattened = len(set(question_mapping.values()))
    logger.info(f"Question flattening: {questions_changed} questions standardized, {len(unique_questions)} -> {unique_flattened} unique questions")
    
    return question_mapping


async def _flatten_answers_with_tracking(qa_pairs: List[QAPair], headers: Dict, logger, results: FlatteningResults) -> Dict[str, str]:
    """Phase 2 with tracking: Create answer mappings and record transformations"""
    
    unique_answers = list(set(qa.answer for qa in qa_pairs))
    logger.info(f"Found {len(unique_answers)} unique answers to analyze")
    
    if len(unique_answers) <= 1:
        answer_mapping = {a: a for a in unique_answers}
        results.answer_mappings = answer_mapping
        return answer_mapping
    
    # Process answers in batches
    batch_size = 20 #tbd use pipeline batch size
    answer_mapping = {}
    
    for i in range(0, len(unique_answers), batch_size):
        batch = unique_answers[i:i + batch_size]
        logger.info(f"Processing answer batch {i//batch_size + 1}/{(len(unique_answers) + batch_size - 1)//batch_size}")
        
        try:
            batch_mapping = await _flatten_answer_batch(batch, headers, logger)
            answer_mapping.update(batch_mapping)
            
            if i + batch_size < len(unique_answers):
                await asyncio.sleep(PipelineConfig.RATE_LIMIT_DELAY)
                
        except Exception as e:
            logger.error(f"Failed to process answer batch {i//batch_size + 1}: {str(e)}")
            for answer in batch:
                answer_mapping[answer] = answer
            continue
    
    # Store answer mappings in results
    results.answer_mappings = answer_mapping
    
    # Count answer standardizations
    answers_changed = sum(1 for orig, flat in answer_mapping.items() if orig != flat)
    unique_flattened = len(set(answer_mapping.values()))
    logger.info(f"Answer flattening: {answers_changed} answers standardized, {len(unique_answers)} -> {unique_flattened} unique answers")
    
    return answer_mapping


async def _flatten_reasons_with_tracking(qa_pairs: List[QAPair], headers: Dict, logger, results: FlatteningResults) -> Dict[str, str]:
    """Phase 3 with tracking: Create reason mappings and record transformations"""
    
    unique_reasons = list(set(qa.reason for qa in qa_pairs))
    logger.info(f"Found {len(unique_reasons)} unique reasons to analyze")
    
    if len(unique_reasons) <= 1:
        reason_mapping = {r: r for r in unique_reasons}
        results.reason_mappings = reason_mapping
        return reason_mapping
    
    # Process reasons in batches
    batch_size = 20 # TBD use pipeline batch size
    reason_mapping = {}
    
    for i in range(0, len(unique_reasons), batch_size):
        batch = unique_reasons[i:i + batch_size]
        logger.info(f"Processing reason batch {i//batch_size + 1}/{(len(unique_reasons) + batch_size - 1)//batch_size}")
        
        try:
            batch_mapping = await _flatten_reason_batch(batch, headers, logger)
            reason_mapping.update(batch_mapping)
            
            if i + batch_size < len(unique_reasons):
                await asyncio.sleep(PipelineConfig.RATE_LIMIT_DELAY)
                
        except Exception as e:
            logger.error(f"Failed to process reason batch {i//batch_size + 1}: {str(e)}")
            for reason in batch:
                reason_mapping[reason] = reason
            continue
    
    # Store reason mappings in results
    results.reason_mappings = reason_mapping
    
    # Count reason standardizations
    reasons_changed = sum(1 for orig, flat in reason_mapping.items() if orig != flat)
    unique_flattened = len(set(reason_mapping.values()))
    logger.info(f"Reason flattening: {reasons_changed} reasons standardized, {len(unique_reasons)} -> {unique_flattened} unique reasons")
    
    return reason_mapping


async def _flatten_question_batch(questions: List[str], headers: Dict, logger) -> Dict[str, str]:
    """Flatten a batch of questions"""
    
    questions_data = [{"index": i, "question": q} for i, q in enumerate(questions)]
    
    prompt = f"""
    Analyze the following questions and identify groups that ask essentially the same thing but with different wording.

    For each group of semantically equivalent questions:
    1. Choose the clearest, most concise version as the standard
    2. Map all similar variants to this standard version
    3. Preserve questions that are truly unique

    Rules:
    - Only group questions that are truly asking the same thing
    - Preserve the exact meaning and intent
    - Choose the most natural, clear wording as the standard
    - Return ALL questions with their standardized version (even if unchanged)

    Input questions:
    {json.dumps(questions_data, indent=2)}

    IMPORTANT: Return ONLY a valid JSON array. No explanations, no additional text.
    Each object must have "original_question" and "standardized_question" fields.
    Example format:
    [
      {{"original_question": "What is X?", "standardized_question": "What is X?"}},
      {{"original_question": "Tell me about X", "standardized_question": "What is X?"}}
    ]
    """
    
    payload = {
        "model": PipelineConfig.LLM_MODEL,
        "max_tokens": PipelineConfig.MAX_TOKENS,
        "temperature": 0.0,
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
            
            result = response.json()
            content = result["content"][0]["text"]
            
            # Parse JSON response
            flattened_data = json.loads(content)

            # Create mapping dictionary
            mapping = {}
            for item in flattened_data:
                original = item.get("original_question", "")
                standardized = item.get("standardized_question", original)
                if original:
                    mapping[original] = standardized
            
            # Ensure all input questions are mapped
            for question in questions:
                if question not in mapping:
                    mapping[question] = question
            
            return mapping
    
    except Exception as e:
        logger.error(f"Failed to flatten question batch: {str(e)}")
        logger.error(f"{str(flattened_data)}")
        logger.error(f"{str(item)}")
        # Return identity mapping as fallback
        return {q: q for q in questions}


async def _flatten_answer_batch(answers: List[str], headers: Dict, logger) -> Dict[str, str]:
    """Flatten a batch of answers"""
    
    answers_data = [{"index": i, "answer": a} for i, a in enumerate(answers)]
    
    prompt = f"""
    Analyze the following answers and identify groups that convey essentially the same information but with different wording.

    For each group of semantically equivalent answers:
    1. Choose the clearest, most complete version as the standard
    2. Map all similar variants to this standard version
    3. Preserve answers that are truly unique

    Rules:
    - Only group answers that convey essentially the same information
    - Preserve the exact meaning and intent
    - Choose the most natural, clear wording as the standard
    - Return ALL answers with their standardized version (even if unchanged)

    Input answers:
    {json.dumps(answers_data, indent=2)}

    Return a JSON array with objects containing "original_answer" and "standardized_answer". 
    Return only the JSON array, no additional text. Make sure it starts and ends with square brackets.

    IMPORTANT: Return ONLY a valid JSON array. No explanations, no additional text.
    Each object must have "original_answer" and "standardized_answer" fields.
    Example format:
    [
      {{"original_answer": "I like X", "standardized_answer": "I like X"}},
      {{"original_answer": "X is great", "standardized_answer": "I like X"}}
    ]
    """
    
    payload = {
        "model": PipelineConfig.LLM_MODEL,
        "max_tokens": PipelineConfig.MAX_TOKENS,
        "temperature": 0.0,
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
            
            result = response.json()
            content = result["content"][0]["text"]
            
            # Parse JSON response
            flattened_data = json.loads(content)

            # Create mapping dictionary
            mapping = {}
            for item in flattened_data:
                original = item.get("original_answer", "")
                standardized = item.get("standardized_answer", original)
                if original:
                    mapping[original] = standardized
            
            # Ensure all input answers are mapped
            for answer in answers:
                if answer not in mapping:
                    mapping[answer] = answer
            
            return mapping
    
    except Exception as e:
        logger.error(f"Failed to flatten answer batch: {str(e)}")
        # Return identity mapping as fallback
        return {a: a for a in answers}


async def _flatten_reason_batch(reasons: List[str], headers: Dict, logger) -> Dict[str, str]:
    """Flatten a batch of reasons"""
    
    reasons_data = [{"index": i, "reason": r} for i, r in enumerate(reasons)]
    
    prompt = f"""
    Analyze the following reasons and identify groups that convey essentially the same information but with different wording.

    For each group of semantically equivalent reasons:
    1. Choose the clearest, most complete version as the standard
    2. Map all similar variants to this standard version
    3. Preserve reasons that are truly unique

    Rules:
    - Only group reasons that convey essentially the same information
    - Preserve the exact meaning and intent
    - Choose the most natural, clear wording as the standard
    - Return ALL reasons with their standardized version (even if unchanged)

    Input reasons:
    {json.dumps(reasons_data, indent=2)}

    Return a JSON array with objects containing "original_reason" and "standardized_reason". 
    Return only the JSON array, no additional text. Make sure it starts and ends with square brackets.
    """
    
    payload = {
        "model": PipelineConfig.LLM_MODEL,
        "max_tokens": PipelineConfig.MAX_TOKENS,
        "temperature": 0.0,
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
            
            result = response.json()
            content = result["content"][0]["text"]
            
            # Parse JSON response
            flattened_data = json.loads(content)

            # Create mapping dictionary
            mapping = {}
            for item in flattened_data:
                original = item.get("original_reason", "")
                standardized = item.get("standardized_reason", original)
                if original:
                    mapping[original] = standardized
            
            # Ensure all input reasons are mapped
            for reason in reasons:
                if reason not in mapping:
                    mapping[reason] = reason
            
            return mapping
    
    except Exception as e:
        logger.error(f"Failed to flatten reason batch: {str(e)}")
        # Return identity mapping as fallback
        return {r: r for r in reasons}


async def _apply_mappings_with_tracking(qa_pairs: List[QAPair], question_mapping: Dict[str, str], answer_mapping: Dict[str, str], reason_mapping: Dict[str, str], results: FlatteningResults):
    """Phase 4: Apply all mappings and create transformations with tracking"""
    
    for qa in qa_pairs:
        # Apply mappings
        flattened_question = question_mapping.get(qa.question, qa.question)
        flattened_answer = answer_mapping.get(qa.answer, qa.answer)
        flattened_reason = reason_mapping.get(qa.reason, qa.reason)
        
        # Create flattened QA pair
        flattened_qa = QAPair(
            question=flattened_question,
            answer=flattened_answer,
            reason=flattened_reason,
            transcript_id=qa.transcript_id
        )
        
        # Determine transformation type
        changes = []
        if qa.question != flattened_question:
            changes.append("question")
        if qa.answer != flattened_answer:
            changes.append("answer")
        if qa.reason != flattened_reason:
            changes.append("reason")
        
        if not changes:
            transformation_type = "unchanged"
            transformation_reason = "No changes needed"
        else:
            transformation_type = f"{'+'.join(changes)}_standardized"
            transformation_reason = f"Standardized: {', '.join(changes)}"
        
        # Add transformation
        results.add_transformation(FlatteningTransformation(
            original_qa=qa,
            flattened_qa=flattened_qa,
            transformation_type=transformation_type,
            transformation_reason=transformation_reason
        ))



# Store dataframes and variables for later human evaluation
@task(name='store_artifacts', retries=1)
def create_prefect_artifacts(
    raw_transcripts: List[TranscriptData],
    cleaned_transcripts: List[TranscriptData], 
    qa_pairs: List[QAPair],
    qa_pairs_flattened: List[QAPair],
    # statistical_results: Dict[str, Any],
    # connection_results: Dict[str, Any],
    # final_results: AnalysisResults
):
    """Store all critical pipeline elements as Prefect artifacts for later evaluation"""
    logger = get_run_logger()
    logger.info("Creating Prefect artifacts for pipeline evaluation")
    
    from prefect.artifacts import create_table_artifact, create_markdown_artifact
    import json
    from datetime import datetime
    
    current_time = datetime.now().isoformat()
    
    # 1. Store Transcript Data Overview
    transcript_overview = []
    for i, (raw, clean) in enumerate(zip(raw_transcripts[:10], cleaned_transcripts[:10])):  # Limit to first 10 for readability
        transcript_overview.append({
            'transcript_id': raw.transcript_id,
            'original_length': len(raw.content),
            'cleaned_length': len(clean.content),
            'metadata': json.dumps(raw.metadata),
            'content_preview': clean.content[:200] + "..." if len(clean.content) > 200 else clean.content
        })
    
    create_table_artifact(
        key="transcript-overview",  # Changed from transcript_overview
        table=transcript_overview,
        description=f"Overview of processed transcripts (showing first 10 of {len(raw_transcripts)} total)"
    )
    
    # 2. Store QA Pairs Data
    qa_pairs_data = []
    for qa in qa_pairs[:50]:  # Limit to first 50 for readability
        qa_pairs_data.append({
            'transcript_id': qa.transcript_id,
            'question': qa.question,
            'answer': qa.answer,
            'reason': qa.reason,
            'question_length': len(qa.question),
            'answer_length': len(qa.answer),
            'reason_length': len(qa.reason)
        })
    
    create_table_artifact(
        key="qa-pairs-sample",  # Changed from qa_pairs_sample
        table=qa_pairs_data,
        description=f"Sample of extracted Q&A pairs (showing first 50 of {len(qa_pairs)} total)"
    )

     # 2.5. Store Flattened QA Pairs Data
    qa_pairs_data_flattened = []
    for qa in qa_pairs_flattened:#[:50]:  # Limit to first 50 for readability
        qa_pairs_data_flattened.append({
            'transcript_id': qa.transcript_id,
            'question': qa.question,
            'answer': qa.answer,
            'reason': qa.reason,
            'question_length': len(qa.question),
            'answer_length': len(qa.answer),
            'reason_length': len(qa.reason)
        })
    
    create_table_artifact(
        key="qa-pairs-flattened-sample",  # Changed from qa_pairs_flattened_sample
        table=qa_pairs_data_flattened,
        description=f"Sample of flattened Q&A pairs (showing first 50 of {len(qa_pairs_flattened)} total)"
    )


# TBD - combine with create_prefect artifacts function and clean up.
@task(name='store_flattening_artifacts', retries=1)
def create_flattening_artifacts(
    flattening_results: FlatteningResults,
    raw_transcripts: List[TranscriptData],
    cleaned_transcripts: List[TranscriptData], 
    qa_pairs: List[QAPair],
    qa_pairs_flattened: List[QAPair],
    # statistical_results: Dict[str, Any],
    # connection_results: Dict[str, Any],
    # final_results: AnalysisResults
):

    logger = get_run_logger()
    logger.info(f"Creating Artifacts for Flattening Evaluation")

    """Create comprehensive artifacts from flattening results"""
    from prefect.artifacts import create_table_artifact, create_markdown_artifact
    import json
    from datetime import datetime
    
    current_time = datetime.now().isoformat()

    # 1. Transformation Summary
    transformation_summary = []
    for i, t in enumerate(flattening_results.transformations[:100]):  # First 100
        transformation_summary.append({
            'index': i,
            'transcript_id': t.original_qa.transcript_id,
            'group_id': t.group_id,
            'transformation_type': t.transformation_type,
            'transformation_reason': t.transformation_reason,
            'original_question': t.original_qa.question,
            'flattened_question': t.flattened_qa.question if t.flattened_qa else "[REMOVED]",
            'original_answer': t.original_qa.answer,
            'flattened_answer': t.flattened_qa.answer if t.flattened_qa else "[REMOVED]",
            'original_reason': t.original_qa.reason,
            'flattened_reason': t.flattened_qa.reason if t.flattened_qa else "[REMOVED]"
        })
    
    create_table_artifact(
        key="flattening-transformations",
        table=transformation_summary,
        description=f"Detailed transformation tracking (showing first 100 of {len(flattening_results.transformations)} transformations)"
    )
    
    # 2.1 Question Mappings
    question_mappings_data = []
    for orig, flat in flattening_results.question_mappings.items():
        question_mappings_data.append({
            'original_question': orig,
            'standardized_question': flat,
            'changed': orig != flat
        })
    
    create_table_artifact(
        key="question-mappings",
        table=question_mappings_data,
        description=f"Question standardization mappings ({len(question_mappings_data)} questions)"
    )
    
    # 2.2 Answer Mappings
    answer_mappings_data = []
    for orig, flat in flattening_results.answer_mappings.items():
        answer_mappings_data.append({
            'original_answer': orig,
            'standardized_answer': flat,
            'changed': orig != flat
        })
    
    create_table_artifact(
        key="answer-mappings",
        table=answer_mappings_data,
        description=f"Answer standardization mappings ({len(answer_mappings_data)} answers)"
    )

    # 2.3 Reason Mappings
    reason_mappings_data = []
    for orig, flat in flattening_results.reason_mappings.items():
        reason_mappings_data.append({
            'original_reason': orig,
            'standardized_reason': flat,
            'changed': orig != flat
        })
    
    create_table_artifact(
        key="reason-mappings",
        table=reason_mappings_data,
        description=f"reason standardization mappings ({len(reason_mappings_data)} reasons)"
    )

    # 3. Statistics Summary
    stats_markdown = f"""
    # Flattening Statistics

    ## Overall Summary
    - **Original Pairs**: {flattening_results.statistics['total_original']:,}
    - **Flattened Pairs**: {flattening_results.statistics['total_flattened']:,}
    - **Groups Created**: {flattening_results.statistics['groups_created']:,}

    ## Transformation Breakdown
    - **Questions Standardized**: {flattening_results.statistics['questions_standardized']:,}
    - **Answers Standardized**: {flattening_results.statistics['answers_standardized']:,}
    - **Reasons Standardized**: {flattening_results.statistics['reasons_standardized']:,}
    - **Pairs Removed**: {flattening_results.statistics['pairs_removed']:,}

    ## Transformation Types
    """
    
    # Count transformation types
    type_counts = {}
    for t in flattening_results.transformations:
        type_counts[t.transformation_type] = type_counts.get(t.transformation_type, 0) + 1
    
    for trans_type, count in sorted(type_counts.items()):
        stats_markdown += f"- **{trans_type}**: {count:,} pairs\n"
    
    create_markdown_artifact(
        key="flattening-statistics",
        markdown=stats_markdown,
        description="Comprehensive flattening process statistics"
    )


    
    # 7. Store Complete Data Export (JSON format for programmatic access)
    complete_export = {
        'metadata': {
            'generated_at': current_time,
            'pipeline_version': '1.0',
            'total_transcripts': len(raw_transcripts),
            'total_qa_pairs': len(qa_pairs)
        },
        'transcript_summary': {
            'total_raw': len(raw_transcripts),
            'total_cleaned': len(cleaned_transcripts),
            'sample_transcript_ids': [t.transcript_id for t in raw_transcripts[:5]]
        },
        'qa_summary': {
            'total_pairs': len(qa_pairs),
            'sample_questions': [qa.question for qa in qa_pairs[:5]],
            'transcripts_with_qa': list(set(qa.transcript_id for qa in qa_pairs))
        },
        # 'analysis_summary': {
        #     'statistical_results': statistical_results,
        #     'connection_results': connection_results,
        #     'final_insights': final_results.insights
        # }
    }
    
    create_markdown_artifact(
        key="complete-data-export",  # Changed from complete_data_export
        markdown=f"```json\n{json.dumps(complete_export, indent=2, default=str)}\n```",
        description="Complete data export in JSON format for programmatic access"
    )
    
    logger.info(f"Successfully created 7 artifacts for pipeline evaluation")
    logger.info(f"Artifacts contain data for {len(raw_transcripts)} transcripts and {len(qa_pairs)} Q&A pairs")
    
    return {
        'artifacts_created': 7,
        'transcripts_stored': len(raw_transcripts),
        'qa_pairs_stored': len(qa_pairs),
        # 'insights_generated': len(final_results.insights)
    }
@task(name="export-data-csv", retries=1)
def export_qa_pairs_to_csv(filename: str, qa_pairs: List[QAPair]):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['question', 'answer', 'reason', 'transcript_id'])  # header
        for qa in qa_pairs:
            writer.writerow([qa.question, qa.answer, qa.reason, qa.transcript_id])

# Main Pipeline Flow
@flow(name="transcript-processing-pipeline", task_runner=ConcurrentTaskRunner())
async def transcript_processing_pipeline(input_file_path: str) -> AnalysisResults:
    """
    Main pipeline flow for processing feedback transcripts
    
    Sequential steps:
    1. Load and clean transcripts
    2. Extract Q&A pairs using LLM
    2.5 Flatten Q&A pairs to standardize similar content
    3. Perform statistical analysis
    4. Model connections between responses
    5. Compile final results
    6. Store artifacts for evaluation
    """
    logger = get_run_logger()
    logger.info("Starting transcript processing pipeline")
    current_time = datetime.now().isoformat()

    # Step 1: Data ingestion and preprocessing
    raw_transcripts = load_transcripts(input_file_path)
    cleaned_transcripts = clean_transcripts(raw_transcripts)
    # TBD store transcripts locally
    # TBD add "ammend transcripts funciton."
    
    # Step 2: LLM processing for Q&A extraction
    qa_pairs_raw = await batch_extract_qa_pairs(cleaned_transcripts)
    export_qa_pairs_to_csv(f'qa_pairs_raw_{current_time}.csv', qa_pairs_raw)
    # Step 2.5: clean up and flatten the QA pairs, so that the same answers have the same wording (allow easier statistical analysis)
    qa_pairs_flattened_results = await flatten_qa_pairs_with_tracking(qa_pairs_raw)
    qa_pairs_flattened = qa_pairs_flattened_results.get_flattened_pairs()
    export_qa_pairs_to_csv(f'qa_pairs_flattenend_{current_time}.csv', qa_pairs_flattened)
    # # Step 3: Statistical analysis
    # statistical_results = perform_statistical_analysis(qa_pairs_flattened)
    
    # # Step 4: Connection modeling
    # connection_results = model_qa_connections(qa_pairs_flattened)
    
    # # Step 5: Compile final results
    
    # final_results = compile_final_results(statistical_results, connection_results)
    
  # Step 6: Store artifacts for evaluation
    create_prefect_artifacts(
        raw_transcripts=raw_transcripts,
        cleaned_transcripts=cleaned_transcripts,
        qa_pairs=qa_pairs_raw,  # Original pairs
        qa_pairs_flattened=qa_pairs_flattened,  # Flattened pairs
        # statistical_results=statistical_results,
        # connection_results=connection_results,
        # final_results=final_results
    )

    create_flattening_artifacts(
        flattening_results=qa_pairs_flattened_results,
        raw_transcripts=raw_transcripts,
        cleaned_transcripts=cleaned_transcripts,
        qa_pairs=qa_pairs_raw,  # Original pairs
        qa_pairs_flattened=qa_pairs_flattened,  # Flattened pairs
        # statistical_results=statistical_results,
        # connection_results=connection_results,
        # final_results=final_results
    )

    # Step 7: Export qa_pairs_flattened to csv, for using in other scripts.
    
    

    logger.info("Pipeline completed successfully")
    # return final_results
    return qa_pairs_flattened_results

# Deployment and Scheduling
if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def run_pipeline():
        # results = await transcript_processing_pipeline("mock_feedback.csv")
        results = await transcript_processing_pipeline("rwth_feedback.csv")
        # # Save results
        # TBD save other dataframes to json

        # with open(f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        #     json.dump({
        #         'stats': results.stats,
        #         'connections': results.connections,
        #         'insights': results.insights,
        #         'generated_at': datetime.now().isoformat()
        #     }, f, indent=2, default=str)
        
        
        print("Pipeline completed successfully!")
        # print(f"Generated {len(results.insights)} insights:")
        # for insight in results.insights:
        #     print(f"- {insight}")
    
    # Run the pipeline
    asyncio.run(run_pipeline())