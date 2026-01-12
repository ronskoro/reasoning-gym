#!/bin/bash
#SBATCH --job-name="reasoning_gym"
#SBATCH --partition=a100-galvani
#SBATCH --time=9:00:00
#SBATCH --gpus=4
#SBATCH --output=/home/bethge/bkr261/.logs/%u-%x-%j.out
#SBATCH --error=/home/bethge/bkr261/.logs/%u-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ronald.skorobogat@bethgelab.org
#SBATCH --nodes=1

~/.conda/envs/reasoning-gym/bin/python -u train_grpo.py \
    --config-path configs/inter_generalisation \
    --config-name algorithmic_qwen_3b \
    trainer.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    data.train_batch_size=8 \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    +trainer.gradient_accumulation_steps=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2
