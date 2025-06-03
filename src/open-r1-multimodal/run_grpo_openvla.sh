PROJECT_ROOT="$(pwd)"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# Change the data_paths and image_folders to your own data
model_path="openvla/openvla-7b"

export EXP_NAME="openvla-rl" # TODO: change this to your own experiment name
# cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}

export WANDB_DISABLED=false
export WANDB_API_KEY=28b3c634497c0dc6c16767729d4719b1012a94f2
export WANDB_PROJECT=openvla-rl
export WANDB_ENTITY=mahlerrrr76

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
  src/open_r1/grpo_openvla.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/runs/${EXP_NAME} \
    --dataset_name this_is_not_used \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 100 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 8 \
    --max_completion_length 200 \
    --reward_funcs token_accuracy \
    --beta 0.04 \
    --report_to wandb \
    --deepspeed local_scripts/zero2.json \
    --vla_path openvla/openvla-7b \
    --data_root_dir /gpfs/yanghan/openvla-mini-o1/dataset/rl_bench_o1_dataset/2.0.0 \
    --vla_dataset_name rlbencho1 \
    --lora_rank 32 \
    --num_images_in_input 2 \
    --use_proprio True \
    --image_aug True

echo "Training completed for ${EXP_NAME}"
