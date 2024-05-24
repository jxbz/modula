if test "$3" = "adam" && test "$4" != 1; then
    LRS=(0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125)
    LRS=(0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125)
else
    LRS=(0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 3.0 4.0)
fi

if test "$3" = "adam"; then
    BETA2=0.99
else
    BETA2=-1
fi

WIDTHS=(32 64 128 256 512 1024)
DEPTH=3

for WIDTH in ${WIDTHS[@]}; do
     GPU=0
     for LR in ${LRS[@]}; do
          TAG=$1/$2/$3/$4/$WIDTH/$DEPTH/$LR
          echo running $TAG
          mkdir -p logs/$TAG
          git archive -o logs/$TAG/code.zip HEAD
          export CUDA_VISIBLE_DEVICES=$GPU
          python main.py --arch $2 --dataset $1 --depth $DEPTH --width $WIDTH --lr $LR --train_steps 10000 --context 128 --wd 0.0 --normalize $4 --beta2 $BETA2 \
              --log_dir logs/$TAG 1> logs/$TAG/out.log 2> logs/$TAG/err.log &
          ((GPU+=1))
     done
     wait
done

DEPTHS=(2 4 8 16 32 64)
WIDTH=128

for DEPTH in ${DEPTHS[@]}; do
     GPU=0
     for LR in ${LRS[@]}; do
          TAG=$1/$2/$3/$4/$WIDTH/$DEPTH/$LR
          echo running $TAG
          mkdir -p logs/$TAG
          git archive -o logs/$TAG/code.zip HEAD
          export CUDA_VISIBLE_DEVICES=$GPU
          python main.py --arch $2 --dataset $1 --depth $DEPTH --width $WIDTH --lr $LR --train_steps 10000 --context 128 --wd 0.0 --normalize $4 --beta2 $BETA2 \
              --log_dir logs/$TAG 1> logs/$TAG/out.log 2> logs/$TAG/err.log &
          ((GPU+=1))
     done
     wait
done
