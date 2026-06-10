#!/bin/bash
# --- Run identification ----------------------------------------------------
export F_HEADER=siouxfalls_run
export SEED=2021
export N_AGENT=20000
export AGENT_TRAJ_STRIDE=1

# --- Common knobs -------------------------------------------------------------
export OBS_STEP=10
export N_STEPS_FORWARD=0
export USE_OBS_FILES=0
export ASSIM_N_ITER=200
export RUN_FORWARD_OPT=0
export FORWARD_TARGET_RATIO=0.5
export FORWARD_OPT_N_ITER=200

export OPTIMIZE_BETA=1
export OPTIMIZE_U=1
export OPTIMIZE_KAPPA=1
export OPTIMIZE_MP=1

export FORCE_ASSIM=1
export DEBUG_VJP=0
export CHUNK_SIZE=9999
export USE_CHECKPOINT=1
export AD_MODE=reverse

export USE_PARAM_REPARAM=1
export PARAM_BETA_SCALE=2.5
export PARAM_U_SCALE=18.05
export PARAM_KAPPA_SCALE=0.2
export PARAM_MP_SCALE=2.5
export PARAM_BETA_MIN=0.0
export PARAM_U_MIN=0.0
export PARAM_KAPPA_MIN=0.15
export PARAM_KAPPA_MAX=0.25
export PARAM_MP_MIN=1e-3

LR=0.1
export ASSIM_LR=$LR
export FORWARD_OPT_LR=$LR
export ASSIM_METHOD=adam
export ADAM_OPTIMIZER=adamw
export N_VMAP_SAMPLES=1
export EPS_SIG=0.1
export MAX_GRAD_NORM=100.0
export GRAD_TOL=1e-8
export ADAM_BETA1=0.9
export ADAM_BETA2=0.999
export ADAM_EPS=1e-8
export ADAM_LR=$LR
export ADAM_LR_SCHEDULE=constant
export ADAM_LR_DECAY_RATE=0.92
export ADAM_LR_DECAY_STEPS=100
export ADAM_LR_WARMUP_STEPS=0
export ADAM_LR_END_VALUE=1e-2
export SNAPSHOT_OFFSET_FRACTION=0.04
export SNAPSHOT_ARROW_SCALE=0.2
export SNAPSHOT_ARROW_LW=2.0

export DELTA_N_OBS=1.0

# delta_n = 1 
export N_STEPS=1800
export CUM_SAVE_STEP=10
export OBS_STEP=300
export DELTA_T=1.0
export DELTA_N=1.0
python3 -u scripts/calibrate_siouxfalls.py

# delta_n = 2
export N_STEPS=900
export CUM_SAVE_STEP=5
export OBS_STEP=150
export DELTA_T=2.0
export DELTA_N=2.0
python3 -u scripts/calibrate_siouxfalls.py

# delta_n = 4
export N_STEPS=450
export CUM_SAVE_STEP=2
export OBS_STEP=75
export DELTA_T=4.0
export DELTA_N=4.0
python3 -u scripts/calibrate_siouxfalls.py