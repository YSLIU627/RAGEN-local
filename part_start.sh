set -e
export RAY_START_WAIT_TIME_S=60
export TOKENIZERS_PARALLELISM=True
export NCCL_DEBUG='WARN'
export VLLM_LOGGING_LEVEL='WARN'
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=84f03efa3815c8727157b1951519ce4b0f2a190a
#export WANDB_ENTITY=RLHF-zhihan

MODEL_PATH='Qwen/Qwen2.5-7B-Instruct'

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae critic.model.path=${MODEL_PATH}" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
# Section 3.1&3.2 - General Observations


PROJECT_NAME='_2_sokoban'
EXPERIMENT_NAME='PPO-Qwen2.5-7B-Instruct'
# Section 4.1 - Filtering and critic
# 0.25
N=3  # 最大的 process 编号

if [ $# -eq 0 ]; then
    CASE=($(seq 1 $N))
else
    CASE=("$@")
fi

if [[ " ${CASE[@]} " =~ " 1 " ]]; then
    TASK_NAME='_1_bandit'
    python train.py --config-name ${TASK_NAME} \
            actor_rollout_ref.model.path=${MODEL_PATH} \
            trainer.experiment_name=sokoban-ppo-rolloutfilter0.5 \
            actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
            data.custom_temp_dir='/fsx-project/zhihan0627/tmp' $USE_PPO \
            trainer.experiment_name=${EXPERIMENT_NAME} \
            trainer.project_name=${TASK_NAME} \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=1
fi
if [[ " ${CASE[@]} " =~ " 2 " ]]; then
TASK_NAME='_2_sokoban'
python train.py --config-name ${TASK_NAME} \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        trainer.experiment_name=sokoban-ppo-rolloutfilter0.5 \
        actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
        data.custom_temp_dir='/fsx-project/zhihan0627/tmp' $USE_PPO \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.project_name=${TASK_NAME} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1
fi

if [[ " ${CASE[@]} " =~ " 3 " ]]; then
TASK_NAME='_3_frozen_lake'
python train.py --config-name ${TASK_NAME} \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        trainer.experiment_name=sokoban-ppo-rolloutfilter0.5 \
        actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
        data.custom_temp_dir='/fsx-project/zhihan0627/tmp' $USE_PPO \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.project_name=${TASK_NAME} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1
fi

if [[ " ${CASE[@]} " =~ " 6 " ]]; then
TASK_NAME='_6_webshop'
python train.py --config-name ${TASK_NAME} \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        trainer.experiment_name=sokoban-ppo-rolloutfilter0.5 \
        actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
        data.custom_temp_dir='/fsx-project/zhihan0627/tmp' $USE_PPO \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.project_name=${TASK_NAME} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1
fi