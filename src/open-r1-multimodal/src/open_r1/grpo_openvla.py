# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch

from open_r1.trainer import OpenVLAGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.vlm_modules import *
from open_r1.grpo_utils import *

from transformers.utils import logging

from openai import OpenAI

logger = logging.get_logger(__name__)


# from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward, monkey_patch_torch_load
# monkey_patch_qwen2_5vl_flash_attn()    
# monkey_patch_torch_load()


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["token_accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

@dataclass
class OpenVLAConfig:
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    vla_dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)

    # Architecture
    num_images_in_input: int = 1                                   # Number of images in the VLA input (default: 1)
    use_proprio: bool = False
    
    # Fine-tuning Parameters
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    resume: bool = False                                            # If True, resumes from checkpoint
    resume_step: Optional[int] = None                               # (When `resume==True`) Step number that we are resuming from

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    vla_lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    
    debug: bool = False


def main(script_args, training_args, model_args, vla_args): 
    if vla_args.debug:
        import pdb
        pdb.set_trace()
    
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Initialize the GRPO trainer
    vla_args.output_dir = training_args.output_dir
    trainer = OpenVLAGRPOTrainer(
        vla_path=vla_args.vla_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vla_args=vla_args,
        attn_implementation=model_args.attn_implementation,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig, OpenVLAConfig))
    script_args, training_args, model_args, vla_args = parser.parse_args_and_config()
    # if training_args.deepspeed and "zero3" in training_args.deepspeed:
    #     print("zero3 is used, qwen2_5vl forward monkey patch is applied")
    #     monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args, vla_args)
