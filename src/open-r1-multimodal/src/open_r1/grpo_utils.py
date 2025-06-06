import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json
import torch

from transformers import AutoProcessor, AutoTokenizer

from open_r1.vlm_modules import *

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)


tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    

def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]

def evaluate_answer_similarity(student_answer, ground_truth):
    """Use llm to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": "You are a evaluation expert. First, analyze the student's response to identify and extract their final answer. Then, compare the extracted answer with the correct solution. Output ONLY '1.0' if the extracted answer matches the correct solution in meaning, or '0.0' if the student's response does not contain a clear or correct answer. No other output is allowed."
                },
                {
                    "role": "user",
                    "content": f"Student's response: {student_answer}\nCorrect solution: {ground_truth}\nOutput only 1.0 or 0.0:"
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return float(result)
    
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if student_answer ==ground_truth else 0.0

def llm_reward(content, sol, **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    return evaluate_answer_similarity(student_answer, ground_truth)

def mcq_reward(content, sol, **kwargs):
    # For multiple choice, extract and compare choices
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    has_choices = extract_choice(ground_truth)
    correct_choice = has_choices.upper() if has_choices else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()
    student_choice = extract_choice(student_answer)
    if student_choice:
        reward = 1.0 if student_choice == correct_choice else 0.0
    else:
        reward = 0.0

    return reward


def yes_no_reward(content, sol, **kwargs):
    content = content.lower()
    sol = sol.lower()

    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()

    ground_yes_no = re.search(r'(yes|no)', ground_truth)
    ground_yes_no = ground_yes_no.group(1) if ground_yes_no else ''
    student_yes_no = re.search(r'(yes|no)', student_answer)
    student_yes_no = student_yes_no.group(1) if student_yes_no else ''

    reward = 1.0 if ground_yes_no == student_yes_no else 0.0

    return reward

# score_type: 0 for mAP, 1 for mAP 50
def calculate_map(pred_bbox_list, gt_bbox_list, score_type=0):
    # Calculate mAP

    # Initialize COCO object for ground truth
    gt_json = {"annotations": [], "images": [], "categories": []}
    gt_json["images"] = [{
        "id": 0,
        "width": 2048,
        "height": 2048,
        "file_name": "image_0.jpg"
    }]

    gt_json["categories"] = []

    cats2id = {}
    cat_count = 0
    for idx, gt_bbox in enumerate(gt_bbox_list):
        if gt_bbox["label"] not in cats2id:
            cats2id[gt_bbox["label"]] = cat_count
            gt_json["categories"].append({
                "id": cat_count,
                "name": gt_bbox["label"]
            })
            cat_count += 1
        
        gt_json["annotations"].append({
            "id": idx+1,
            "image_id": 0,
            "category_id": cats2id[gt_bbox["label"]],
            "bbox": [gt_bbox["bbox_2d"][0], gt_bbox["bbox_2d"][1], gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0], gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1]],
            "area": (gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0]) * (gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1]),
            "iscrowd": 0
        })
    coco_gt = COCO(gt_json)

    dt_json = []
    for idx, pred_bbox in enumerate(pred_bbox_list):
        try:
            dt_json.append({
                "image_id": 0,
                "category_id": cats2id[pred_bbox["label"]],
                "bbox": [pred_bbox["bbox_2d"][0], pred_bbox["bbox_2d"][1], pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0], pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1]],
                "score": 1.0,
                "area": (pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0]) * (pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1])
            })
        except:
            pass
    
    if len(dt_json) == 0:
        return 0.0
    
    coco_dt = coco_gt.loadRes(dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[score_type]

def map_reward(content, sol, length_reward=False, score_type=0, **kwargs):
    """
    Calculate mean average precision (mAP) reward between predicted and ground truth bounding boxes.
    
    Args:
        content (str): String containing predicted bounding boxes in JSON format
        sol (str): String containing ground truth bounding boxes in JSON format
        length_reward (bool, optional): Whether to include length penalty in reward calculation. Defaults to False.
        score_type (int, optional): Type of COCO evaluation metric to use. Defaults to 0 (mAP).
        **kwargs: Additional keyword arguments
        
    Returns:
        float: mAP reward score between 0 and 1. If length_reward is True, the score is multiplied by a length penalty factor.
    """
    # Extract JSON content between ```json tags
    pattern = r'```json(.*?)```'
    json_match = re.findall(pattern, sol, re.DOTALL)
    bbox_json = json_match[-1].strip() if json_match else None

    # Parse ground truth JSON to get bbox list
    gt_bbox_list = []
    if bbox_json:
        bbox_data = json.loads(bbox_json)
        gt_bbox_list = [item for item in bbox_data]
    
    # Parse predicted JSON to get bbox list
    pred_bbox_list = []
    json_match = re.findall(pattern, content, re.DOTALL)
    if json_match:
        try:
            bbox_data = json.loads(json_match[-1].strip())
            pred_bbox_list = [item for item in bbox_data]
        except:
            # Return empty list if JSON parsing fails
            pred_bbox_list = []

    # Calculate mAP if both prediction and ground truth exist
    if len(pred_bbox_list) > 0 and len(gt_bbox_list) > 0:
        bbox_reward = calculate_map(pred_bbox_list, gt_bbox_list, score_type=score_type)
    elif len(pred_bbox_list) == 0 and len(gt_bbox_list) == 0:
        bbox_reward = 1.0
    else:
        bbox_reward = 0.0
    
    if length_reward:
        # Calculate length penalty based on ratio of ground truth to predicted bounding boxes
        gt_length = len(gt_bbox_list)
        pred_length = len(pred_bbox_list)
        # Full score if prediction has fewer boxes than ground truth, otherwise penalize proportionally
        length_score = 1.0 if gt_length >= pred_length else gt_length/pred_length
        return bbox_reward * length_score
    else:
        return bbox_reward

def od_reward(content, sol, score_type=0, **kwargs):
    """
    Calculate reward for object detection task by comparing predicted and ground truth answers.
    
    Args:
        content (str): Model's predicted answer containing bounding box annotations
        sol (str): Ground truth answer containing bounding box annotations 
        score_type (int): Type of COCO evaluation metric to use (default: 0 for mAP)
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Reward score between 0 and 1 based on mAP between predicted and ground truth boxes
    """
    # Pattern to extract content between <answer> tags
    match_pattern = r'<answer>(.*?)</answer>'

    # Extract ground truth answer
    sol_match = re.search(match_pattern, sol, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else None

    # Extract predicted answer (using last match if multiple)
    content_match = re.findall(match_pattern, content, re.DOTALL)
    student_answer = content_match[-1].strip() if content_match else None

    # Return 0 if no prediction
    if student_answer is None:
        return 0.0
    # Return 1 if both prediction and ground truth are None
    elif ground_truth == "None" and student_answer == "None":
        return 1.0
    # Otherwise calculate mAP between prediction and ground truth
    else:
        return map_reward(student_answer, ground_truth, score_type=score_type)

def odLength_reward(content, sol, **kwargs):
    """
    Calculate reward for object detection task with length penalty.
    
    Args:
        content (str): Model's predicted answer containing bounding box annotations
        sol (str): Ground truth answer containing bounding box annotations
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Reward score between 0 and 1 based on mAP and length penalty
    """
    # Pattern to extract content between <answer> tags
    match_pattern = r'<answer>(.*?)</answer>'

    # Extract ground truth answer
    sol_match = re.search(match_pattern, sol, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else None
    # Extract predicted answer (using last match if multiple)
    content_match = re.findall(match_pattern, content, re.DOTALL)
    student_answer = content_match[-1].strip() if content_match else None

    # Return 0 if no prediction
    if student_answer is None:
        return 0.0
    # Return 1 if both prediction and ground truth are None
    elif ground_truth == "None" and student_answer == "None":
        return 1.0
    # Calculate mAP with length penalty
    else:
        bbox_reward = map_reward(student_answer, ground_truth, length_reward=True, score_type=0)
        return bbox_reward

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union


def detection_score(content, sol, iou_threshold=0.5, alpha=0.7, beta=0.0, gamma=0.3):
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, clean_text(content), re.DOTALL)
    content_bbox_json = json_match.group(1).strip() if json_match else None
    if content_bbox_json:
        try:
            bbox_data = json.loads(content_bbox_json)
            pred_boxes = [item for item in bbox_data]
        except:
            pred_boxes = []

    else:
        pred_boxes = []

    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, clean_text(sol), re.DOTALL)
    sol_bbox_json = json_match.group(1).strip() if json_match else None
    if sol_bbox_json:
        bbox_data = json.loads(sol_bbox_json)
        gt_boxes = [item for item in bbox_data]
    else:
        gt_boxes = []

    """
    Calculate the comprehensive score for object detection
    
    Parameters:
        pred_boxes: List of predicted boxes, each element is in the format {"bbox_2d": [x1, y1, x2, y2], "label": "category name"}
        gt_boxes: List of ground truth boxes, each element is in the format {"bbox_2d": [x1, y1, x2, y2], "label": "category name"}
        iou_threshold: IoU threshold, default is 0.5
        alpha: Position accuracy weight, default is 0.7
        beta: Label accuracy weight, default is 0.0
        gamma: Completeness weight (penalty for missed/false detections), default is 0.3
        
    Returns:
        Comprehensive score, ranging from [0.0, 1.0]
    """
    # Handle edge cases
    if len(gt_boxes) == 0:
        return 1.0 if not pred_boxes else 0.0
    
    if len(pred_boxes) == 0:
        return 0.0
    
    # Initialize matching results
    matches = []  # Store matched pairs of predicted and ground truth boxes
    unmatched_preds = list(range(len(pred_boxes)))  # Indices of unmatched predicted boxes
    unmatched_gts = list(range(len(gt_boxes)))  # Indices of unmatched ground truth boxes
    
    # Calculate IoU matrix between all predicted and ground truth boxes
    iou_matrix = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        iou_row = []
        for gt_idx, gt_box in enumerate(gt_boxes):
            try:
                curr_iou = iou(pred_box["bbox_2d"], gt_box["bbox_2d"])
            except:
                curr_iou = 0.0
            iou_row.append(curr_iou)
        iou_matrix.append(iou_row)
    
    # Greedy matching: find the best match for each predicted box
    while unmatched_preds and unmatched_gts:
        # Find the maximum IoU
        max_iou = -1
        max_pred_idx = -1
        max_gt_idx = -1
        
        for pred_idx in unmatched_preds:
            for gt_idx in unmatched_gts:
                curr_iou = iou_matrix[pred_idx][gt_idx]
                if curr_iou > max_iou:
                    max_iou = curr_iou
                    max_pred_idx = pred_idx
                    max_gt_idx = gt_idx
        
        # Stop matching if the maximum IoU is below the threshold
        if max_iou < iou_threshold:
            break
        
        # Record matching results
        try:
            pred_label = pred_boxes[max_pred_idx]["label"].lower()
        except:
            pred_box = ""
        try:
            gt_label = gt_boxes[max_gt_idx]["label"].lower()
        except:
            gt_label = ""
        label_correct = (pred_label == gt_label)
        
        if label_correct:
            matches.append({
                "pred_idx": max_pred_idx,
                "gt_idx": max_gt_idx,
                "iou": max_iou,
                "label_correct": label_correct
            })
        else:
            matches.append({
                "pred_idx": max_pred_idx,
                "gt_idx": max_gt_idx,
                "iou": 0,
                "label_correct": label_correct
            })
        
        # Remove matched boxes from the unmatched list
        unmatched_preds.remove(max_pred_idx)
        unmatched_gts.remove(max_gt_idx)
    
    # Calculate position accuracy score (average IoU)
    position_score = sum(m["iou"] for m in matches) / len(gt_boxes) if matches else 0.0
    
    # Calculate label accuracy score
    label_score = sum(1.0 for m in matches if m["label_correct"]) / len(gt_boxes) if matches else 0.0
    
    # Calculate completeness score (considering missed and false detections)
    # Miss rate = number of unmatched ground truth boxes / total number of ground truth boxes
    # False alarm rate = number of unmatched predicted boxes / total number of predicted boxes
    miss_rate = len(unmatched_gts) / len(gt_boxes)
    false_alarm_rate = len(unmatched_preds) / len(pred_boxes) if pred_boxes else 0.0
    
    # Completeness score = 1 - (miss rate + false alarm rate) / 2
    completeness_score = 1.0 - (miss_rate + false_alarm_rate) / 2.0
    
    # Calculate the final comprehensive score
    final_score = (
        alpha * position_score + 
        beta * label_score + 
        gamma * completeness_score
    ) / (alpha + beta + gamma)

    return final_score

def cosine_reward(content, tokenizer, acc_reward, **kwargs):
    #https://arxiv.org/abs/2502.03373
    min_len_value_wrong = 0.0
    max_len_value_wrong = -0.5
    min_len_value_correct = 1.0
    max_len_value_correct = 0.5
    cosine_max_len = 1024

    # processing_class = AutoProcessor.from_pretrained(model_path)
    # tokenizer = processing_class.tokenizer
    
    gen_len = len(tokenizer.encode(content))
    acc_reward = 1.0
    is_correct = acc_reward >= 0.7
    
    if is_correct:
        # Swap min/max for correct answers
        min_value = max_len_value_correct
        max_value = min_len_value_correct
    else:
        min_value = min_len_value_wrong
        max_value = max_len_value_wrong

    reward = max_value - (max_value - min_value) * (1 - math.cos(gen_len * math.pi / cosine_max_len)) / 2

    return reward

def repetition_reward(content, **kwargs):
    max_penalty = -1.0

    if content == '':
        return 0.0

    # First, try to extract explicitly marked JSON sections
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, content, re.DOTALL)
    
    if json_match:
        bbox_json = json_match.group(1).strip()
    else:
        # If no explicitly marked JSON is found, try to find any possible JSON sections
        pattern = r'```(.*?)```'
        json_match = re.search(pattern, content, re.DOTALL)
        bbox_json = json_match.group(1).strip() if json_match else None
        
        # If still not found, try to find possible JSON array sections
        if not bbox_json:
            pattern = r'\[\s*{.*?"bbox_2d".*?"label".*?}\s*\]'
            json_match = re.search(pattern, content, re.DOTALL)
            bbox_json = json_match.group(0) if json_match else None
    
    # Try to parse JSON data
    if bbox_json:
        try:
            # Try direct parsing
            data = json.loads(bbox_json)
        except json.JSONDecodeError:
            try:
                # If direct parsing fails, try using json_repair to repair
                repaired_json = repair_json(bbox_json)
                data = json.loads(repaired_json)
            except:
                # If repair also fails, switch to plain text processing
                data = None
        if data and isinstance(data, list):
            # Ensure data is in list format
            try:
                # For JSON data, set ngram_size to 1
                ngram_size = 1
                # Combine 'bbox_2d' and 'label' of each object into a string
                items = []
                for item in data:
                    if 'bbox_2d' in item and 'label' in item:
                        items.append(f"{item['bbox_2d']}_{item['label']}")
                
                @staticmethod
                def zipngram(text: list, ngram_size: int):
                    return zip(*[text[i:] for i in range(ngram_size)])
                
                ngrams = set()
                total = 0

                for ng in zipngram(items, ngram_size):
                    ngrams.add(ng)
                    total += 1

                if total == 0:
                    return 0.0

                scaling = 1 - len(ngrams) / total
                reward = scaling * max_penalty

                return reward
            except KeyError:
                # If necessary keys are missing, switch to plain text processing
                pass
    
    # If no JSON section is found or JSON processing fails, treat as plain text
    ngram_size = 6
    
    if len(content.split()) < ngram_size:
        return 0.0
    
    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    
    ngrams = set()
    total = 0

    for ng in zipngram(content, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty

    return reward


def repetition_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        reward = repetition_reward(content)
        rewards.append(reward)


        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 0.0:  # this condition can be changed for debug
                with open(log_path+"_repetition.txt", "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")     



    return rewards


def cosine_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        clean_content = clean_text(content)
        sol = clean_text(sol)
        if sol == "none":
            if clean_content == "none":
                acc_reward = 1.0
            else:
                acc_reward = 0.0
        else:
            acc_reward = detection_score(clean_content, sol)
        reward = cosine_reward(content, tokenizer, acc_reward)
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <=1.0:  # this condition can be changed for debug
                with open(log_path+"_cosine.txt", "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")   

    return rewards

def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None
def math_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return compute_score(content, sol)
def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

def all_match_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return 1.0 if content == sol else 0.0

def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
        # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    
    # Try symbolic verification first for numeric answers
    try:
        answer = parse(student_answer)
        if float(verify(answer, parse(ground_truth))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try: 
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r'\d', ground_truth))
            # Check if it's a multiple choice question
            has_choices = extract_choice(ground_truth)
            
            if has_numbers:
                # For numeric answers, use exact matching
                reward = numeric_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                # For multiple choice, extract and compare choices
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            else:
                # For text answers, use fuzzy matching
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method == "mcq":
            reward = mcq_reward(content, sol)
        elif accu_reward_method == 'yes_no':
            reward = yes_no_reward(content, sol)
        elif accu_reward_method == 'llm':
            reward = llm_reward(content, sol)
        elif accu_reward_method == 'map':
            reward = map_reward(content, sol)
        elif accu_reward_method == 'math':
            reward = math_reward(content, sol)
        elif accu_reward_method == 'weighted_sum':
            clean_content = clean_text(content)
            sol = clean_text(sol)
            if sol == "none":
                if clean_content == "none":
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = detection_score(clean_content, sol)
        elif accu_reward_method == 'od_ap':
            reward = od_reward(content, sol)
        elif accu_reward_method == 'od_ap50':
            reward = od_reward(content, sol, score_type=1)
        elif accu_reward_method == 'odLength':
            reward = odLength_reward(content, sol)
        elif accu_reward_method == 'all_match':
            reward = all_match_reward(content, sol)
        else:
            reward = default_accuracy_reward(content, sol)  
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 1.0:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"accu_reward_method: {accu_reward_method}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")     

        
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]


def token_accuracy_reward(pred_ids, gt_ids, action_tokenizer, action_dim=7, action_weight_ratio=[3, 3, 3, 2, 2, 2, 2], scale_factor=32):
    action_weight_ratio = torch.tensor(action_weight_ratio)
    action_weight_ratio = action_weight_ratio / action_weight_ratio.sum()
    action_weight_ratio = action_weight_ratio.to(pred_ids.device)
    assert action_weight_ratio.shape[0] == action_dim, f"action_weight_ratio should have {action_dim} elements, but got {action_weight_ratio.shape[0]}"
    
    rewards = []
    for batch_idx, (pred, gt) in enumerate(zip(pred_ids, gt_ids)):
        pred_action_ids = pred[(pred > action_tokenizer.action_token_begin_idx) & (pred != action_tokenizer.tokenizer.pad_token_id)]
        gt_action_ids = gt[gt > action_tokenizer.action_token_begin_idx]
        
        if pred_action_ids.shape != gt_action_ids.shape:
            rewards.append(0.0)
            continue
        
        assert len(pred_action_ids) == action_dim, f"pred_action_ids should have {action_dim} elements, but got {len(pred_action_ids)}"
        
        reward = (pred_action_ids - gt_action_ids).abs().float()
        reward = reward / scale_factor
        reward = torch.clamp(reward, 0.0, 1.0)
        reward = 1.0 - reward
        reward = torch.dot(reward, action_weight_ratio)
        rewards.append(reward.item())

    return rewards


def openvla_format_reward(pred_ids, gt_ids, action_tokenizer):
    responses = action_tokenizer.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # gts = action_tokenizer.tokenizer.batch_decode(gt_ids, skip_special_tokens=True)
    
    rewards = []
    for response in responses:
        score = 1.0
        if "ACTION:" not in response:
            score = 0.0
        if "ACTION SUCCESS:" not in response:
            score = 0.0
        if "CURRENT GOAL:" in response:
            judgement = response.split("CURRENT GOAL:")[0].split("ACTION SUCCESS:")[-1].strip()
            if judgement not in ["True", "False"]:
                score = 0.0
        elif "FAILURE REASON:" in response:
            judgement = response.split("FAILURE REASON:")[0].split("ACTION SUCCESS:")[-1].strip()
            if judgement not in ["True", "False"]:
                score = 0.0
        else:
            score = 0.0
        
        rewards.append(score)
        
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": cosine_rewards,
    "repetition": repetition_rewards,
    "token_accuracy": token_accuracy_reward,
    "openvla_format": openvla_format_reward,
}