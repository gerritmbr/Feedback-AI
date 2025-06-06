import asyncio
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import httpx
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.blocks.system import Secret
import networkx as nx
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
        self.transformation_type = transformation_type  # "unchanged", "question_standardized", "answer_standardized", "both_standardized", "removed"
        self.transformation_reason = transformation_reason
        self.group_id = group_id  # For tracking which pairs were grouped together

class FlatteningResults:
    def __init__(self):
        self.transformations: List[FlatteningTransformation] = []
        self.question_mappings: Dict[str, str] = {}  # original -> standardized
        self.statistics = {
            'total_original': 0,
            'total_flattened': 0,
            'questions_standardized': 0,
            'answers_standardized': 0,
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

    Transcript:
    {transcript.content}

    Return only the JSON array, no additional text. Make sure that the response can be parsed directly, so it starts and ends with curled backets.
    """

    payload = {
        "model": PipelineConfig.LLM_MODEL,
        "max_tokens": PipelineConfig.MAX_TOKENS,
        "temperature": PipelineConfig.TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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
            await asyncio.sleep(5)
    
    logger.info(f"Extracted total of {len(all_qa_pairs)} Q&A pairs")
    return all_qa_pairs



# Step 2.5: clean up and flatten the QA pairs, so that the same answers have the same wording (allow easier statistical analysis)
# needs checks

# Modified flattening function with direct storage
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
        
        # Phase 2: Group and flatten answers/reasons with tracking
        logger.info("Phase 2: Flattening answers and reasons per question group with tracking")
        await _flatten_answers_reasons_with_tracking(qa_pairs, question_mapping, headers, logger, results)
        
        # Update final statistics
        results.statistics['total_flattened'] = len(results.get_flattened_pairs())
        results.statistics['questions_standardized'] = sum(1 for t in results.transformations 
                                                         if 'question' in t.transformation_type)
        results.statistics['answers_standardized'] = sum(1 for t in results.transformations 
                                                       if 'answer' in t.transformation_type)
        
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
    batch_size = 100
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

    Return a JSON array with objects containing "original_question" and "standardized_question". 
    Return only the JSON array, no additional text. Make sure it starts and ends with square brackets.
    """
    
    payload = {
        "model": PipelineConfig.LLM_MODEL,
        "max_tokens": PipelineConfig.MAX_TOKENS,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
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
        # Return identity mapping as fallback
        return {q: q for q in questions}




async def _flatten_answers_reasons_with_tracking(qa_pairs: List[QAPair], question_mapping: Dict[str, str], headers: Dict, logger, results: FlatteningResults):
    """Phase 2 with tracking: Group by questions and flatten answers/reasons"""
    
    # Group QA pairs by their standardized question
    question_groups = {}
    for qa in qa_pairs:
        standardized_question = question_mapping.get(qa.question, qa.question)
        if standardized_question not in question_groups:
            question_groups[standardized_question] = []
        question_groups[standardized_question].append(qa)
    
    logger.info(f"Grouped pairs into {len(question_groups)} question groups")
    results.statistics['groups_created'] = len(question_groups)
    
    # Process each question group
    group_counter = 0
    for standardized_question, group_pairs in question_groups.items():
        group_id = f"group_{group_counter}"
        group_counter += 1
        
        if len(group_pairs) <= 1:
            # Single pair - just update question and track
            original_qa = group_pairs[0]
            flattened_qa = QAPair(
                question=standardized_question,
                answer=original_qa.answer,
                reason=original_qa.reason,
                transcript_id=original_qa.transcript_id
            )
            
            transformation_type = "question_standardized" if original_qa.question != standardized_question else "unchanged"
            
            results.add_transformation(FlatteningTransformation(
                original_qa=original_qa,
                flattened_qa=flattened_qa,
                transformation_type=transformation_type,
                transformation_reason=f"Single pair in group, question {'standardized' if transformation_type == 'question_standardized' else 'unchanged'}",
                group_id=group_id
            ))
            continue
        
        logger.info(f"Processing group {group_id} with {len(group_pairs)} pairs for question: '{standardized_question[:50]}...'")
        
        try:
            # Flatten answers and reasons within this group
            await _flatten_group_with_tracking(standardized_question, group_pairs, headers, logger, results, group_id)
            
            # Rate limiting between groups
            await asyncio.sleep(PipelineConfig.RATE_LIMIT_DELAY * 0.5)
            
        except Exception as e:
            logger.error(f"Failed to process group {group_id}: {str(e)}")
            # Fall back to original pairs with updated question
            for pair in group_pairs:
                flattened_qa = QAPair(
                    question=standardized_question,
                    answer=pair.answer,
                    reason=pair.reason,
                    transcript_id=pair.transcript_id
                )
                
                results.add_transformation(FlatteningTransformation(
                    original_qa=pair,
                    flattened_qa=flattened_qa,
                    transformation_type="question_standardized",
                    transformation_reason=f"Fallback due to group processing error: {str(e)}",
                    group_id=group_id
                ))
            continue

#

async def _flatten_group_with_tracking(standardized_question: str, group_pairs: List[QAPair], headers: Dict, logger, results: FlatteningResults, group_id: str):
    """Flatten a group while tracking all transformations"""
    
    # Prepare data for the group
    group_data = []
    for idx, qa in enumerate(group_pairs):
        group_data.append({
            "index": idx,
            "transcript_id": qa.transcript_id,
            "answer": qa.answer,
            "reason": qa.reason
        })
    
    # LLM processing (same as before)
    prompt = f"""
    For the question: "{standardized_question}"

    Analyze the following answer-reason pairs and standardize those that convey essentially the same information but with different wording.

    For each group of semantically equivalent answers or reasons:
    1. Choose the clearest, most complete version as the standard
    2. Replace similar variants with this standard version
    3. Preserve the original transcript_id and index for each pair

    Rules:
    - Only group answers/reasons that convey essentially the same information
    - Preserve meaning exactly - don't change the intent or specificity
    - Keep all pairs in the output with the same structure
    - Maintain the exact same array length

    Input data:
    {json.dumps(group_data, indent=2)}

    Return the same structure with standardized wording where appropriate. 
    Return only the JSON array, no additional text. Make sure it starts and ends with square brackets.
    """
    
    payload = {
        "model": PipelineConfig.LLM_MODEL,
        "max_tokens": PipelineConfig.MAX_TOKENS,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(PipelineConfig.LLM_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["content"][0]["text"]
            flattened_data = json.loads(content)
            
            # Validate response length
            if len(flattened_data) != len(group_pairs):
                logger.warning(f"Group flattening returned {len(flattened_data)} items, expected {len(group_pairs)}. Using fallback.")
                raise ValueError("Response length mismatch")
            
            # Create transformations with detailed tracking
            for item in flattened_data:
                original_idx = item.get("index", 0)
                if original_idx < len(group_pairs):
                    original_qa = group_pairs[original_idx]
                    
                    flattened_answer = item.get("answer", original_qa.answer)
                    flattened_reason = item.get("reason", original_qa.reason)
                    
                    flattened_qa = QAPair(
                        question=standardized_question,
                        answer=flattened_answer,
                        reason=flattened_reason,
                        transcript_id=item.get("transcript_id", original_qa.transcript_id)
                    )
                    
                    # Determine transformation type
                    changes = []
                    if original_qa.question != standardized_question:
                        changes.append("question")
                    if original_qa.answer != flattened_answer:
                        changes.append("answer")
                    if original_qa.reason != flattened_reason:
                        changes.append("reason")
                    
                    if not changes:
                        transformation_type = "unchanged"
                        transformation_reason = "No changes needed"
                    else:
                        transformation_type = f"{'+'.join(changes)}_standardized"
                        transformation_reason = f"Standardized: {', '.join(changes)}"
                    
                    results.add_transformation(FlatteningTransformation(
                        original_qa=original_qa,
                        flattened_qa=flattened_qa,
                        transformation_type=transformation_type,
                        transformation_reason=transformation_reason,
                        group_id=group_id
                    ))
                else:
                    logger.warning(f"Invalid index {original_idx} in group flattened data")
    
    except Exception as e:
        logger.error(f"Failed to flatten group {group_id}: {str(e)}")
        raise


# # Step 3: Statistical Analysis
# # needs to be rewritten -  TBD
# @task(name="perform_statistical_analysis", retries=1)
# def perform_statistical_analysis(qa_pairs: List[QAPair]) -> Dict[str, Any]:
#     """Perform statistical analysis on extracted Q&A data"""
#     logger = get_run_logger()
#     logger.info("Performing statistical analysis")
    
#     # Convert to DataFrame for analysis
#     data = []
#     for qa in qa_pairs:
#         data.append({
#             'transcript_id': qa.transcript_id,
#             'question': qa.question,
#             'answer': qa.answer,
#             'reason': qa.reason,
#             'question_length': len(qa.question), #remove
#             'answer_length': len(qa.answer),  #remove
#             'reason_length': len(qa.reason)  #remove
#         })
    
#     df = pd.DataFrame(data)
    
#     stats = {
#         'total_qa_pairs': len(qa_pairs),
#         'unique_transcripts': df['transcript_id'].nunique(),
#         'avg_qa_pairs_per_transcript': len(qa_pairs) / df['transcript_id'].nunique(),
#         'question_stats': {
#             'avg_length': df['question_length'].mean(),
#             'median_length': df['question_length'].median(),
#             'std_length': df['question_length'].std()
#         },
#         'answer_stats': {
#             'avg_length': df['answer_length'].mean(),
#             'median_length': df['answer_length'].median(),
#             'std_length': df['answer_length'].std()
#         },
#         'most_common_question_patterns': _extract_question_patterns(df['question'].tolist()),
#         'answer_sentiment_distribution': _analyze_answer_sentiment(df['answer'].tolist())
#     }
    
#     logger.info("Statistical analysis completed")
#     return stats

# def _extract_question_patterns(questions: List[str]) -> Dict[str, int]:
#     """Extract common question patterns"""
#     patterns = []
#     for q in questions:
#         # Simple pattern extraction based on question starters
#         q_lower = q.lower().strip()
#         if q_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
#             patterns.append(q_lower.split()[0])
#         elif '?' in q:
#             patterns.append('general_question')
#         else:
#             patterns.append('statement')
    
#     return dict(Counter(patterns).most_common(10))

# def _analyze_answer_sentiment(answers: List[str]) -> Dict[str, int]:
#     """Basic sentiment analysis of answers"""
#     positive_words = ['good', 'great', 'excellent', 'satisfied', 'happy', 'positive', 'yes']
#     negative_words = ['bad', 'poor', 'terrible', 'unsatisfied', 'unhappy', 'negative', 'no']
    
#     sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
#     for answer in answers:
#         answer_lower = answer.lower()
#         pos_count = sum(1 for word in positive_words if word in answer_lower)
#         neg_count = sum(1 for word in negative_words if word in answer_lower)
        
#         if pos_count > neg_count:
#             sentiment_counts['positive'] += 1
#         elif neg_count > pos_count:
#             sentiment_counts['negative'] += 1
#         else:
#             sentiment_counts['neutral'] += 1
    
#     return sentiment_counts

# # Step 4: Connection Modeling
# # needs checks
# @task(name="model_connections", retries=1)
# def model_qa_connections(qa_pairs: List[QAPair]) -> Dict[str, Any]:
#     """Model connections between answers and responses within transcripts"""
#     logger = get_run_logger()
#     logger.info("Modeling Q&A connections")
    
#     # Group by transcript
#     transcript_groups = {}
#     for qa in qa_pairs:
#         if qa.transcript_id not in transcript_groups:
#             transcript_groups[qa.transcript_id] = []
#         transcript_groups[qa.transcript_id].append(qa)
    
#     connection_data = {
#         'transcript_networks': {},
#         'global_patterns': {},
#         'correlation_matrix': {}
#     }
    
#     # Analyze connections within each transcript
#     for transcript_id, qa_list in transcript_groups.items():
#         if len(qa_list) < 2:
#             continue
        
#         # Create network graph for this transcript
#         G = nx.Graph()
        
#         # Add nodes (questions)
#         for i, qa in enumerate(qa_list):
#             G.add_node(i, question=qa.question, answer=qa.answer)
        
#         # Add edges based on answer similarity or sequential connection
#         for i in range(len(qa_list)):
#             for j in range(i+1, len(qa_list)):
#                 # Simple connection based on shared keywords
#                 similarity = _calculate_answer_similarity(qa_list[i].answer, qa_list[j].answer)
#                 if similarity > 0.3:  # Threshold for connection
#                     G.add_edge(i, j, weight=similarity)
        
#         # Calculate network metrics
#         connection_data['transcript_networks'][transcript_id] = {
#             'node_count': G.number_of_nodes(),
#             'edge_count': G.number_of_edges(),
#             'density': nx.density(G) if G.number_of_nodes() > 1 else 0,
#             'clustering_coefficient': nx.average_clustering(G) if G.number_of_edges() > 0 else 0
#         }
    
#     # Calculate global patterns
#     all_answers = [qa.answer.lower() for qa in qa_pairs]
#     connection_data['global_patterns'] = {
#         'common_answer_themes': _extract_common_themes(all_answers),
#         'answer_length_correlation': _calculate_length_correlations(qa_pairs)
#     }
    
#     logger.info("Connection modeling completed")
#     return connection_data

# def _calculate_answer_similarity(answer1: str, answer2: str) -> float:
#     """Calculate simple similarity between two answers"""
#     words1 = set(answer1.lower().split())
#     words2 = set(answer2.lower().split())
    
#     if not words1 or not words2:
#         return 0.0
    
#     intersection = words1.intersection(words2)
#     union = words1.union(words2)
    
#     return len(intersection) / len(union) if union else 0.0

# def _extract_common_themes(answers: List[str]) -> Dict[str, int]:
#     """Extract common themes from answers"""
#     # Simple keyword-based theme extraction
#     all_words = []
#     for answer in answers:
#         words = [word.lower() for word in answer.split() if len(word) > 3]
#         all_words.extend(words)
    
#     return dict(Counter(all_words).most_common(20))

# def _calculate_length_correlations(qa_pairs: List[QAPair]) -> Dict[str, float]:
#     """Calculate correlations between question/answer/reason lengths"""
#     question_lengths = [len(qa.question) for qa in qa_pairs]
#     answer_lengths = [len(qa.answer) for qa in qa_pairs]
#     reason_lengths = [len(qa.reason) for qa in qa_pairs]
    

#     corr_qa = np.corrcoef(question_lengths, answer_lengths)[0, 1]
#     return {
#         'question_answer_corr': corr_qa if not np.isnan(corr_qa) else 0.0,
#         'question_answer_corr': np.corrcoef(question_lengths, answer_lengths)[0, 1],
#         'answer_reason_corr': np.corrcoef(answer_lengths, reason_lengths)[0, 1],
#         'question_reason_corr': np.corrcoef(question_lengths, reason_lengths)[0, 1]
#     }

# # Step 5: Results Compilation
# @task(name="compile_results", retries=1)
# def compile_final_results(stats: Dict[str, Any], connections: Dict[str, Any]) -> AnalysisResults:
#     """Compile final analysis results"""
#     logger = get_run_logger()
#     logger.info("Compiling final results")
    
#     # Generate insights based on analysis
#     insights = []
    
#     # Statistical insights
#     if stats['avg_qa_pairs_per_transcript'] > 5:
#         insights.append("High engagement: Transcripts contain multiple question-answer exchanges")
    
#     if stats['answer_sentiment_distribution']['positive'] > stats['answer_sentiment_distribution']['negative']:
#         insights.append("Overall positive sentiment in responses")
    
#     # Connection insights
#     avg_density = np.mean([net['density'] for net in connections['transcript_networks'].values()])
#     if avg_density > 0.5:
#         insights.append("Strong interconnections between answers within transcripts")
    
#     results = AnalysisResults(
#         stats=stats,
#         connections=connections,
#         insights=insights
#     )
    
#     logger.info(f"Analysis completed with {len(insights)} key insights")
#     return results

# Store dataframes and variables for later human evaluation
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
    for qa in qa_pairs_flattened[:50]:  # Limit to first 50 for readability
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
    
    # 2. Question Mappings
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


    # # 3. Store Statistical Results Summary
    # stats_markdown = f"""
    # # Statistical Analysis Results
    # *Generated at: {current_time}*

    # ## Overview
    # - **Total Q&A Pairs**: {statistical_results['total_qa_pairs']:,}
    # - **Unique Transcripts**: {statistical_results['unique_transcripts']:,}
    # - **Average Q&A Pairs per Transcript**: {statistical_results['avg_qa_pairs_per_transcript']:.2f}

    # ## Question Statistics
    # - **Average Length**: {statistical_results['question_stats']['avg_length']:.1f} characters
    # - **Median Length**: {statistical_results['question_stats']['median_length']:.1f} characters
    # - **Standard Deviation**: {statistical_results['question_stats']['std_length']:.1f} characters

    # ## Answer Statistics
    # - **Average Length**: {statistical_results['answer_stats']['avg_length']:.1f} characters
    # - **Median Length**: {statistical_results['answer_stats']['median_length']:.1f} characters
    # - **Standard Deviation**: {statistical_results['answer_stats']['std_length']:.1f} characters

    # ## Question Patterns
    # """
    
    # for pattern, count in statistical_results['most_common_question_patterns'].items():
    #     stats_markdown += f"- **{pattern}**: {count} occurrences\n"
    
    # stats_markdown += "\n## Answer Sentiment Distribution\n"
    # for sentiment, count in statistical_results['answer_sentiment_distribution'].items():
    #     stats_markdown += f"- **{sentiment.capitalize()}**: {count} answers\n"
    
    # create_markdown_artifact(
    #     key="statistical-analysis",  # Changed from statistical_analysis
    #     markdown=stats_markdown,
    #     description="Complete statistical analysis of Q&A pairs"
    # )
    
    # # 4. Store Connection Analysis Results
    # connection_markdown = f"""
    # # Connection Analysis Results
    # *Generated at: {current_time}*

    # ## Transcript Network Analysis
    # """
    
    # if connection_results['transcript_networks']:
    #     total_networks = len(connection_results['transcript_networks'])
    #     avg_density = sum(net['density'] for net in connection_results['transcript_networks'].values()) / total_networks
        
    #     connection_markdown += f"""
    # - **Total Transcript Networks**: {total_networks}
    # - **Average Network Density**: {avg_density:.3f}

    # ### Network Details (First 10 Transcripts)
    # """
        
    #     for i, (transcript_id, network_data) in enumerate(list(connection_results['transcript_networks'].items())[:10]):
    #         connection_markdown += f"""
    # #### Transcript {transcript_id}
    # - Nodes: {network_data['node_count']}
    # - Edges: {network_data['edge_count']}
    # - Density: {network_data['density']:.3f}
    # - Clustering Coefficient: {network_data['clustering_coefficient']:.3f}
    # """
    
    # connection_markdown += "\n## Global Patterns\n"
    
    # if 'common_answer_themes' in connection_results['global_patterns']:
    #     connection_markdown += "### Common Answer Themes\n"
    #     for theme, count in list(connection_results['global_patterns']['common_answer_themes'].items())[:10]:
    #         connection_markdown += f"- **{theme}**: {count} occurrences\n"
    
    # if 'answer_length_correlation' in connection_results['global_patterns']:
    #     correlations = connection_results['global_patterns']['answer_length_correlation']
    #     connection_markdown += f"""
    # ### Length Correlations
    # - **Question ↔ Answer**: {correlations.get('question_answer_corr', 0):.3f}
    # - **Answer ↔ Reason**: {correlations.get('answer_reason_corr', 0):.3f}
    # - **Question ↔ Reason**: {correlations.get('question_reason_corr', 0):.3f}
    # """
    
    # create_markdown_artifact(
    #     key="connection-analysis",  # Changed from connection_analysis
    #     markdown=connection_markdown,
    #     description="Analysis of connections and patterns in Q&A data"
    # )
    
    #     # 5. Store Final Results and Insights
    #     insights_markdown = f"""
    #     # Final Analysis Results & Insights
    #     *Generated at: {current_time}*

    #     ## Key Insights
    #     """
        
    #     for i, insight in enumerate(final_results.insights, 1):
    #         insights_markdown += f"{i}. {insight}\n"
        
    #     insights_markdown += f"""

    #     ## Summary Statistics
    #     - **Processing Date**: {current_time}
    #     - **Total Transcripts Processed**: {len(raw_transcripts)}
    #     - **Total Q&A Pairs Extracted**: {len(qa_pairs)}
    #     - **Analysis Insights Generated**: {len(final_results.insights)}

    #     ## Data Quality Metrics
    #     - **Transcripts Successfully Cleaned**: {len(cleaned_transcripts)} / {len(raw_transcripts)} ({len(cleaned_transcripts)/len(raw_transcripts)*100:.1f}%)
    #     - **Average Content Reduction**: {(1 - sum(len(t.content) for t in cleaned_transcripts) / sum(len(t.content) for t in raw_transcripts)) * 100:.1f}%
    #     """
    
    # create_markdown_artifact(
    #     key="final-insights",  # Changed from final_insights
    #     markdown=insights_markdown,
    #     description="Final analysis results and key insights"
    # )
    
    #     # 6. Store Raw Data Counts for Verification
    #     verification_data = [{
    #         'metric': 'Raw Transcripts Loaded',
    #         'count': len(raw_transcripts),
    #         'details': f"From input file processing"
    #     }, {
    #         'metric': 'Cleaned Transcripts',
    #         'count': len(cleaned_transcripts),
    #         'details': f"After preprocessing and validation"
    #     }, {
    #         'metric': 'Total Q&A Pairs Extracted',
    #         'count': len(qa_pairs),
    #         'details': f"From LLM processing"
    #     }, {
    #         'metric': 'Unique Transcript IDs in Q&A',
    #         'count': len(set(qa.transcript_id for qa in qa_pairs)),
    #         'details': f"Transcripts that generated Q&A pairs"
    #     }, {
    #         'metric': 'Statistical Analysis Patterns',
    #         'count': len(statistical_results.get('most_common_question_patterns', {})),
    #         'details': f"Question patterns identified"
    #     }, {
    #         'metric': 'Network Connections Analyzed',
    #         'count': len(connection_results.get('transcript_networks', {})),
    #         'details': f"Transcript networks created"
    #     }]
    
    # create_table_artifact(
    #     key="verification-metrics",  # Changed from verification_metrics
    #     table=verification_data,
    #     description="Verification metrics for pipeline data processing"
    # )
    
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
    
    # Step 1: Data ingestion and preprocessing
    raw_transcripts = load_transcripts(input_file_path)
    cleaned_transcripts = clean_transcripts(raw_transcripts)
    
    # Step 2: LLM processing for Q&A extraction
    qa_pairs_raw = await batch_extract_qa_pairs(cleaned_transcripts)

    # Step 2.5: clean up and flatten the QA pairs, so that the same answers have the same wording (allow easier statistical analysis)
    qa_pairs_flattened_results = await flatten_qa_pairs_with_tracking(qa_pairs_raw)
    qa_pairs_flattened = qa_pairs_flattened_results.get_flattened_pairs()
    
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
    export_qa_pairs_to_csv('qa_pairs.csv', qa_pairs_flattened)

    logger.info("Pipeline completed successfully")
    # return final_results
    return qa_pairs_flattened_results

# Deployment and Scheduling
if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def run_pipeline():
        results = await transcript_processing_pipeline("mock_feedback.csv")
        
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