from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from huggingface_hub import HfApi, snapshot_download

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from open_r1.vlm_modules.vlm_module import VLMBaseModule
from open_r1.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from open_r1.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from open_r1.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from open_r1.prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from open_r1.prismatic.models.backbones.llm.prompting import PurePromptBuilder
from open_r1.prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from open_r1.prismatic.vla.constants import IGNORE_INDEX
from open_r1.prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from open_r1.prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from open_r1.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from open_r1.prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from open_r1.prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from open_r1.prismatic.training.finetune_utils import *
from open_r1.prismatic.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

    

class OpenVLAModule(VLMBaseModule):
    def __init__(self, cfg):
        super().__init__()
        
        assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
        assert not (cfg.use_l1_regression and cfg.use_diffusion), (
            "Cannot do both L1 regression and diffusion. Please pick one of them!"
        )
        
        # GPU setup
        distributed_state = PartialState()
        device_id = distributed_state.local_process_index
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        
        # Trim trailing forward slash ('/') in VLA path if it exists
        cfg.vla_path = cfg.vla_path.rstrip("/")
        print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
        
        # Two options:
        # (1) Base model is on Hugging Face Hub
        #   - Then download it and record the path to the download directory
        # (2) Base model is stored locally
        #   - Then register model config in HF Auto Classes
        # In both cases, we want to check whether any changes have been made to
        # the `modeling_prismatic.py` file in this codebase; if so, we will copy
        # the file to the downloaded or locally stored checkpoint directory so
        # that the user's changes to the VLA class logic go into effect
        if model_is_on_hf_hub(cfg.vla_path):
            # Download model directly from Hugging Face Hub
            vla_download_path = snapshot_download(repo_id=cfg.vla_path)
            # Overwrite VLA path
            cfg.vla_path = vla_download_path
        else:
            # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
        # Update config.json and sync model files
        if distributed_state.is_main_process:
            update_auto_map(cfg.vla_path)
            check_model_logic_mismatch(cfg.vla_path)

        # Wait for model files to be synced
        dist.barrier()
        
        # Load processor and VLA
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device_id)

        # Set number of images in VLA input
        vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)


    
    def get_vlm_key(self):
        return "openvla"
    
    