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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized
from pathlib import Path

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
# from trl import GRPOTrainer

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from open_r1.vlm_modules.vlm_module import VLMBaseModule
from open_r1.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from open_r1.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from open_r1.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from open_r1.prismatic.models.backbones.llm.prompting import PurePromptBuilder
from open_r1.prismatic.models.projectors import ProprioProjector
from open_r1.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from open_r1.prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from open_r1.prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from open_r1.prismatic.training.finetune_utils import *

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def remove_prompt_logps(
    per_token_logps: torch.Tensor, prompt_lengths: torch.Tensor, completion_lengths: torch.Tensor,
    prompt_completion_ids: torch.Tensor, pad_token_id: int
):
    all_per_token_logps = []
    for i in range(per_token_logps.shape[0]):
        all_per_token_logps.append(
            per_token_logps[i, prompt_lengths[i] - 1 : prompt_lengths[i] - 1 + completion_lengths[i]]
        )
    pad_value = per_token_logps[prompt_completion_ids[:, 1:] == pad_token_id].mean().item()
    all_per_token_logps = torch.nn.utils.rnn.pad_sequence(all_per_token_logps, batch_first=True, padding_value=pad_value)
    return all_per_token_logps


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class OpenVLAGRPOTrainer(Trainer):
    def __init__(
        self,
        vla_path: str,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vla_args = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
    ):
        # Args
        if args is None:
            args = GRPOConfig(f"{vla_path}-GRPO")
        
        # Trim trailing forward slash ('/') in VLA path if it exists
        print(f"Fine-tuning OpenVLA Model `{vla_args.vla_path}` on `{vla_args.vla_dataset_name}`")
        
        # GPU setup
        distributed_state = PartialState()
        
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Load OpenVLA Processor and Model using HF AutoClasses
        processor = AutoProcessor.from_pretrained(vla_args.vla_path, trust_remote_code=True)
        vla = OpenVLAForActionPrediction.from_pretrained(
            vla_args.vla_path,
            torch_dtype=torch.bfloat16 if torch_dtype == "bfloat16" else torch.float32,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # Set number of images in VLA input
        vla.vision_backbone.set_num_images_in_input(vla_args.num_images_in_input)
        
        self.use_cache = False if args.gradient_checkpointing else args.use_cache

        # LoRA setup
        if vla_args.use_lora:
            lora_config = LoraConfig(
                r=vla_args.lora_rank,
                lora_alpha=min(vla_args.lora_rank, 16),
                lora_dropout=vla_args.vla_lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()
            
            model_ref = vla.model
        else:
            model_ref = vla

        # If applicable, instantiate proprio projector
        if vla_args.use_proprio:
            proprio_projector = init_module(
                ProprioProjector,
                "proprio_projector",
                cfg=vla_args,
                module_args={"llm_dim": model_ref.llm_dim, "proprio_dim": 8},
                to_bf16=True,
                warp_ddp=False
            )
            model_ref.proprio_projector = proprio_projector
        
        # Get number of vision patches
        NUM_PATCHES = model_ref.vision_backbone.get_num_patches() * model_ref.vision_backbone.get_num_images_in_input()
        # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
        if vla_args.use_proprio:
            NUM_PATCHES += 1
        
        # TODO: check whether proprio_projector is in trainable parameters
        
        # Create Action Tokenizer
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)
        
        # Compute the number of trainable parameters and print the parameter that is trainable
        trainable_params = [p for p in model_ref.parameters() if p.requires_grad]
        total_trainable_params = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in model_ref.parameters())
        print(f"Total trainable parameters: {total_trainable_params:,} out of {total_params:,} in the model.")

        # Enable gradient checkpointing if requested
        # TODO: check whether this part is implemented
        if args.gradient_checkpointing:
            vla = self._enable_gradient_checkpointing(vla, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForVision2Seq.from_pretrained(
                vla_args.vla_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        elif is_peft_model(vla):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(vla)
        
        pad_token_id = processor.tokenizer.pad_token_id
        processor.pad_token_id = pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        # Reward processing class
        # TODO
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes
        
        # Init dataset
        
        # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
        use_wrist_image = vla_args.num_images_in_input > 1
        
        # Create training and optional validation datasets
        batch_transform = RLDSBatchTransform(
            self.action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=use_wrist_image,
            use_proprio=vla_args.use_proprio,
        )
        train_dataset = RLDSDataset(
            vla_args.data_root_dir,
            vla_args.vla_dataset_name,
            batch_transform,
            resize_resolution=tuple(model_ref.config.image_sizes),
            shuffle_buffer_size=vla_args.shuffle_buffer_size,
            image_aug=vla_args.image_aug,
        )
        val_dataset = RLDSDataset(
            vla_args.data_root_dir,
            vla_args.vla_dataset_name,
            batch_transform,
            resize_resolution=tuple(model_ref.config.image_sizes),
            shuffle_buffer_size=vla_args.shuffle_buffer_size // 10,
            image_aug=vla_args.image_aug,
            train=False,
        )
        
        # [Important] Save dataset statistics so that we can unnormalize actions during inference
        if distributed_state.is_main_process:
            save_dataset_statistics(train_dataset.dataset_statistics, Path(args.output_dir))
        
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="left"
        )

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
        )
        self.generation_config.eos_token_id = processor.tokenizer.eos_token_id
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        vla.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=vla,
            args=args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=processor,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            # if self.is_deepspeed_enabled:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            # if getattr(model, "language_model", None) is not None:
            #     # For InternVL; these operations are copied from the original training script of InternVL
            #     model.language_model.config.use_cache = False
            #     model.vision_model.gradient_checkpointing = True
            #     model.vision_model.encoder.gradient_checkpointing = True
            #     model.language_model._set_gradient_checkpointing()
            #     # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
            #     args.gradient_checkpointing = False
            # else:
            #     model.gradient_checkpointing_enable()
            
            assert isinstance(model, OpenVLAForActionPrediction), "Only OpenVLAForActionPrediction supports gradient checkpointing"
            
            if hasattr(model.language_model.config, "use_cache"):
                model.language_model.config.use_cache = False
            if hasattr(model.language_model, "_set_gradient_checkpointing"):
                model.language_model._set_gradient_checkpointing(True)
            elif hasattr(model.language_model, "gradient_checkpointing_enable"):
                model.language_model.gradient_checkpointing_enable()
            else:
                raise ValueError(f"The model {model.__class__.__name__} does not support gradient checkpointing.")
            
            if hasattr(model.vision_backbone.featurizer, "grad_checkpointing"):
                model.vision_backbone.featurizer.grad_checkpointing = True
            # Some timm models might use a setter method
            elif hasattr(model.vision_backbone.featurizer, "set_grad_checkpointing"):
                model.vision_backbone.featurizer.set_grad_checkpointing(True)
            else:
                raise ValueError(f"The vision backbone {model.vision_backbone.featurizer.__class__.__name__} does not support gradient checkpointing.")
            
            # For the fused featurizer, if it exists
            if hasattr(model.vision_backbone, "fused_featurizer") and model.vision_backbone.fused_featurizer is not None:
                if hasattr(model.vision_backbone.fused_featurizer, "grad_checkpointing"):
                    model.vision_backbone.fused_featurizer.grad_checkpointing = True
                elif hasattr(model.vision_backbone.fused_featurizer, "set_grad_checkpointing"):
                        model.vision_backbone.fused_featurizer.set_grad_checkpointing(True)
                else:
                    raise ValueError(f"The fused featurizer {model.vision_backbone.fused_featurizer.__class__.__name__} does not support gradient checkpointing.")

            # This line is to prevent the Trainer from trying to enable it again using a generic method,
            # as we've handled it specifically for OpenVLA.
            args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        input_ids, prompt_mask = inputs["input_ids"], inputs["attention_mask"]
        pixel_values, proprio = inputs["pixel_values"].to(torch.bfloat16), inputs["proprio"].to(torch.bfloat16)

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:            
            proprio_projector = unwrapped_model.proprio_projector
            # generated_ids = unwrapped_model.generate(
            #     input_ids=input_ids,
            #     attention_mask=prompt_mask,
            #     pixel_values=pixel_values,
            #     proprio=proprio,
            #     proprio_projector=proprio_projector,
            #     generation_config=self.generation_config,
            #     use_cache=self.use_cache,
            # )
            
            # use loop to iterate each item in the batch
            all_generated_ids = []
            prompt_lengths = prompt_mask.sum(dim=1)
            completion_lengths = []
            for batch_i in range(input_ids.shape[0]):
                input_ids_i = input_ids[batch_i:batch_i + 1]
                prompt_mask_i = prompt_mask[batch_i:batch_i + 1]
                # remove the paddings
                pad_length = prompt_mask_i.shape[-1] - prompt_mask_i.sum().item()
                input_ids_i = input_ids_i[:, pad_length:]
                prompt_mask_i = prompt_mask_i[:, pad_length:]
                generated_id = unwrapped_model.generate(
                    input_ids=input_ids_i,
                    attention_mask=prompt_mask_i,
                    pixel_values=pixel_values[batch_i:batch_i + 1],
                    proprio=proprio[batch_i:batch_i + 1],
                    proprio_projector=proprio_projector,
                    generation_config=self.generation_config,
                    use_cache=self.use_cache,
                )
                all_generated_ids.append(generated_id)
                completion_lengths.append(generated_id.shape[1] - input_ids_i.shape[1])
            max_gen_length = max(generated_id.shape[1] for generated_id in all_generated_ids)
            # Pad the generated_ids to the max_gen_length
            prompt_completion_ids = torch.full(
                (len(all_generated_ids), max_gen_length),
                fill_value=self.processing_class.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            for i, generated_id in enumerate(all_generated_ids):
                prompt_completion_ids[i, :generated_id.shape[1]] = generated_id

        # Mask everything after the first EOS token
        is_eos = prompt_completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        attention_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # get the completion mask
        completion_mask = torch.zeros(
            (len(completion_lengths), max(completion_lengths)),
            dtype=torch.long,
            device=device
        )
        for i, completion_length in enumerate(completion_lengths):
            completion_mask[i, :completion_length] = 1

        # Get the multimodal inputs
        multimodal_inputs = {"pixel_values": pixel_values, "proprio": proprio, "proprio_projector": proprio_projector}
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                # old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
                old_per_token_logps = remove_prompt_logps(
                    old_per_token_logps, prompt_lengths, completion_lengths, 
                    prompt_completion_ids, self.processing_class.pad_token_id
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
        if ref_per_token_logps is not None:
            # ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            ref_per_token_logps = remove_prompt_logps(
                ref_per_token_logps, prompt_lengths, completion_lengths, 
                prompt_completion_ids, self.processing_class.pad_token_id
            )

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(input_ids), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            output_reward_func = reward_func(prompt_completion_ids, inputs["labels"], self.action_tokenizer)
            print(f"{reward_func.__name__} output: {output_reward_func}")
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        
        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(input_ids),
            (self.accelerator.process_index + 1) * len(input_ids),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_completion_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "prompt_lenghts": prompt_lengths,
            "completion_mask": completion_mask,
            "completion_lengths": completion_lengths,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        # prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        # completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = inputs["prompt_completion_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_lengths = inputs["prompt_lenghts"]
        completion_mask = inputs["completion_mask"]
        completion_lengths = inputs["completion_lengths"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        # # Concatenate for full sequence
        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        # per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]
        per_token_logps = remove_prompt_logps(
            per_token_logps, prompt_lengths, completion_lengths, 
            input_ids, self.processing_class.pad_token_id
        )

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        
        # Log the value of trainable parameters to see whether the weights are updated
        mean_values = 0.0
        for name, param in model.base_model.named_parameters():
            if param.requires_grad:
                mean_values += param.abs().mean()
        self._metrics["mean_trainable_param_value"].append(self.accelerator.gather_for_metrics(mean_values).mean().item())

        return loss

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        assert self.is_deepspeed_enabled, "Checkpoint saving is only supported with DeepSpeed enabled."
        unwrapped_model = self.accelerator.unwrap_model(self.deepspeed)
        
        proprio_projector = unwrapped_model.proprio_projector if hasattr(unwrapped_model, "proprio_projector") else None
        if proprio_projector is not None:
            torch.save(
                proprio_projector.state_dict(),
                os.path.join(output_dir, "proprio_projector--checkpoint.pt")
            )
                

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
