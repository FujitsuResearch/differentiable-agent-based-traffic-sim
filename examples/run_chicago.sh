#!/bin/bash
# --- Run identification ----------------------------------------------------
export F_HEADER=chicago_run
export SEED=1234

# --- Agent / time discretisation -------------------------------------------
export AGENT_TRAJ_STRIDE=100
export N_AGENT=1000030
export N_STEPS=60
export N_STEPS_FORWARD=120
export OBS_STEP=10
export DELTA_T=30.0
export DELTA_N=30.0
export DELTA_N_OBS=30.0
export DELTA_T_OBS=30.0
export CUM_SAVE_STEP=1

# --- Data assimilation -----------------------------------------------------
export USE_OBS_FILES=0
export ASSIM_N_ITER=200
export RUN_FORWARD_OPT=1
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

# --- Re-parametrisation ----------------------------------------------------
export USE_PARAM_REPARAM=1
export PARAM_BETA_SCALE=1.0
export PARAM_U_SCALE=1.0
export PARAM_KAPPA_SCALE=1.0
export PARAM_MP_SCALE=1.0
export PARAM_BETA_MIN=0.0
export PARAM_U_MIN=0.0
export PARAM_KAPPA_MIN=0.15
export PARAM_KAPPA_MAX=0.25
export PARAM_MP_MIN=1e-3

# --- Optimiser -------------------------------------------------------------
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

# --- LR scheduling ---------------------------------------------------------
# Schedule type: constant | exponential | cosine | polynomial
export ADAM_LR_SCHEDULE=constant
export ADAM_LR_DECAY_RATE=0.92
export ADAM_LR_DECAY_STEPS=100
export ADAM_LR_WARMUP_STEPS=0
export ADAM_LR_END_VALUE=5e-4

# --- Visualisation knobs ---------------------------------------------------
export SNAPSHOT_OFFSET_FRACTION=0.04
export SNAPSHOT_ARROW_SCALE=0.2
export SNAPSHOT_ARROW_LW=2.0

# --- Run -------------------------------------------------------------------
python3 -u scripts/calibrate_chicago.py