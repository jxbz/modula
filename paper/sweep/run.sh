#!/bin/bash

#SBATCH --output=/dev/null
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20

mkdir -p logs/$TAG
git archive -o logs/$TAG/code.zip HEAD

source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

export OMP_NUM_THREADS=20

python main.py \
  --log_dir logs/$TAG \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \
  --test_steps $TEST_STEPS \
  --dataset $DATASET \
  --arch $ARCH \
  --depth $DEPTH \
  --block_depth $BLOCK_DEPTH \
  --width $WIDTH \
  --context $CONTEXT \
  --num_heads $NUM_HEADS \
  --normalize $NORMALIZE \
  --loss $LOSS \
  --lr $LR \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --wd $WD \
  1> logs/$TAG/out.log \
  2> logs/$TAG/err.log
