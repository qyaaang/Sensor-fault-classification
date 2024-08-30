
# Default arguments
DATASET="IPC_SHM"
FEATURE="hist"
BATCH=200
D_FF=128
DROPOUT=0
N_ATT=4
N_SUBLAYERS=2
N_ENCODERS=4
RANDOM_SEED=0
DATA_AUG=true
SOFTMAX=false

if [[ $DATASET == "IPC_SHM" ]]; then
  N_CLASS=7
  SEQ_LEN=72000
  WIN_LEN=2000
  HIST_BIN=64
  HEAD=8
fi

if [[ $DATASET == "Canton_Tower" ]]; then
  N_CLASS=5
  SEQ_LEN=180000
  WIN_LEN=2000
  HIST_BIN=64
  HEAD=8
fi

if [[ $DATASET == "UCSD_Acc" ]]; then
  N_CLASS=5
  SEQ_LEN=70920
  WIN_LEN=2000
  HIST_BIN=64
  HEAD=8
fi

if [[ $DATASET == "UCSD_Disp" ]]; then
  N_CLASS=5
  SEQ_LEN=70920
  WIN_LEN=2000
  HIST_BIN=64
  HEAD=8
fi

if [[ $DATASET == "UCSD" ]]; then
  N_CLASS=5
  SEQ_LEN=70920
  WIN_LEN=2000
  HIST_BIN=64
  HEAD=8
fi

if [[ $DATASET == "HIT-dataset" ]]; then
  N_CLASS=4
  SEQ_LEN=81920
  WIN_LEN=2560
  HIST_BIN=64
  HEAD=8
fi

python main.py \
  -dataset $DATASET \
  -feature $FEATURE \
  -batch $BATCH \
  -seq_len $SEQ_LEN \
  -win_len $WIN_LEN \
  -hist_bin $HIST_BIN \
  -d_ff $D_FF \
  -n_class $N_CLASS \
  -dropout $DROPOUT \
  -head $HEAD \
  -n_att $N_ATT \
  -n_sublayers $N_SUBLAYERS \
  -n_encoders $N_ENCODERS \
  -random_seed $RANDOM_SEED \
  $([[ $DATA_AUG == true ]] && echo "-data_aug") \
  $([[ $SOFTMAX == true ]] && echo "-softmax")

