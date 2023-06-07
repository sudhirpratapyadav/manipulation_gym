#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=OpenManipulatorPick headless=False seed=${SEED} \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=PPO \
task.env.object.type=cuboid_default \
train.ppo.priv_info=True train.ppo.proprio_adapt=False \
train.ppo.output_name=OpenManipulationPick/"${CACHE}" \
${EXTRA_ARGS}