#!/bin/bash
# FedSHA on the modality x organ federation (ISBI). Place in system/ (next to main.py).
#   ./run.sh <run> [gpu] [name] [extra args...]
#       name (default <run>) = log file logs_mo/<name>.log and the run's goal (-go); extra args are appended.
#   ./run.sh <group> "<gpus>"      two runs per GPU, e.g. ./run.sh baselines "0 1"
#
#   ours:             fedsha (sliding window + gate-informed aggregation) | ft (FedSHA-P, FT_FROM=<goal>, default fedsha)
#   global baselines: central | fedavg (FedMSA) | fedprox | moon | fedmenu | fedmosa  (one global model)
#   personalized:     local | ditto | fedrep | fedhca2 | fedala | feddbe  (best model per client; local output layer)
#   ablations:        no_window | no_gate | organ_only
#   test_all [gpu] [runs]: test the saved best checkpoints into one log (logs_mo/test_all.log)
#   dump_all [gpu] [runs]: save predictions on fixed test slices (qualitative figure)
#   migrate:          rename checkpoints / results of the old code (model name atmoe, runs gate_eq,
#                     no_usage_proto, feddbe_alt) to the new names (fedsha, fedsha, no_gate, feddbe)
# Logs: logs_mo/<name>.log ; follow with tail -f logs_mo/<name>.log
# Each run sees only its own GPU (CUDA_VISIBLE_DEVICES), hence "-gpu 0".
#
# Clients (modorgan6): 0 LiTS (CT liver) | 1 DLDS (MR liver) | 2 KiTS23 (CT kidney) | 3 AMOS22 (CT spleen)
#                      4 BraTS (MR brain) | 5 BHSD (CT brain)

SAM_CKPT="./medsam_vit_b.pth"
GPU=0
ROUNDS=25
BATCH=16
SLICES_PER_EPOCH=2500
LOG_DIR="logs_mo"
mkdir -p "$LOG_DIR"

COMMON="-data modorgan6 -nc 6 --sam_ckpt $SAM_CKPT -lbs $BATCH -spe $SLICES_PER_EPOCH"

launch () {   # launch <name> <gpu> <args...>
    local name=$1 gpu=$2; shift 2
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 -u main.py $COMMON -gpu 0 -go "$name" "$@" > "$LOG_DIR/$name.log" 2>&1 &
    echo "started $name on GPU $gpu (pid $!) -> $LOG_DIR/$name.log"
}

args_for () {
    case "$1" in
        # ours
        fedsha)       echo "-gr $ROUNDS" ;;
        ft)           echo "--ft_from ${FT_FROM:-fedsha} --ft_epochs 10 --select client" ;;   # FedSHA-P
        # ablations of FedSHA
        no_window)    echo "-gr $ROUNDS --hard_window 0 --expert_agg fedavg" ;;   # no window: top-k of the whole grid, FedAvg
        no_gate)      echo "-gr $ROUNDS --expert_agg window" ;;                   # n_c over the clients with the adapter in their window
        organ_only)   echo "-gr $ROUNDS --win_stages 0,12" ;;                     # organ region in all 12 blocks
        # baselines (common base network: dense MSA adapters)
        local)        echo "-gr $ROUNDS --arch local -algo Local --select client" ;;
        central)      echo "-gr $ROUNDS --arch central -algo Centralized" ;;
        fedavg)       echo "-gr $ROUNDS --arch fedavg -algo FedMSA" ;;
        fedprox)      echo "-gr $ROUNDS --arch fedprox -algo FedProx" ;;
        moon)         echo "-gr $ROUNDS --arch moon -algo MOON" ;;
        fedmenu)      echo "-gr $ROUNDS --arch fedmenu -algo FedMENU" ;;
        fedmosa)      echo "-gr $ROUNDS --arch fedmosa -algo FedMoSA" ;;
        ditto)        echo "-gr $ROUNDS --arch ditto -algo Ditto --select client" ;;
        fedrep)       echo "-gr $ROUNDS --arch fedrep -algo FedRep --select client --shared_head 0" ;;
        fedhca2)      echo "-gr $ROUNDS --arch fedhca2 -algo FedHCA2 --select client --shared_head 0" ;;
        fedala)       echo "-gr $ROUNDS --arch fedala -algo FedALA --select client --shared_head 0" ;;
        feddbe)       echo "-gr $ROUNDS --arch feddbe -algo FedDBE --select client --shared_head 0" ;;
        *)            echo "" ;;
    esac
}

run_one () {   # run_one <run> <gpu> [name] [extra args...]
    local run=$1 gpu=$2 name=${3:-$1}
    local a; a=$(args_for "$run")
    [ -z "$a" ] && { echo "unknown run '$run'"; exit 1; }
    shift 3 2>/dev/null || shift $#
    [ -f "$LOG_DIR/$name.log" ] && echo "note: $LOG_DIR/$name.log exists and will be overwritten"
    launch "$name" "$gpu" $a "$@"
}

run_group () {   # two runs per GPU, filled in order
    local gpus=($1); shift
    [ ${#gpus[@]} -eq 0 ] && { echo "give the GPUs, e.g. ./run.sh ablations \"0 1 2\""; exit 1; }
    local i=0
    for run in "$@"; do
        run_one "$run" "${gpus[$(( (i / 2) % ${#gpus[@]} ))]}"
        i=$((i + 1))
    done
}

BASELINES="central fedavg fedprox moon fedmenu fedmosa"
PERSONAL="local ditto fedrep fedhca2 fedala feddbe"   # + ft (FedSHA-P, after the FedSHA run)
ABLATIONS="no_window no_gate organ_only"
TEST_RUNS="fedsha $BASELINES $PERSONAL ft $ABLATIONS"

loop_runs () {   # loop_runs <gpu> <log> <extra args> <runs...>: one run after the other into one log
    local gpu=$1 log=$2 extra=$3; shift 3
    local runs="$*"
    : > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" nohup bash -c '
        for run in '"$runs"'; do
            a=$(bash '"$0"' __args "$run")
            echo "### $run" >> '"$log"'
            python3 -u main.py '"$COMMON"' -gpu 0 -go "$run" $a '"$extra"' >> '"$log"' 2>&1 || echo "### $run FAILED" >> '"$log"'
        done
        echo "### done" >> '"$log"'
        grep "SUMMARY" '"$log"' > '"${log%.log}"'_summary.txt' > /dev/null 2>&1 &
    echo "[$runs] on GPU $gpu (pid $!) -> $log"
}

migrate () {   # rename the files of the old code to the new model / run names (safe to run twice)
    local d f g dirs="../checkpoints ../results ../results/cases ../results/preds $LOG_DIR"
    if [ -n "$(find $dirs -maxdepth 1 -name '*feddbe_alt*' 2>/dev/null | head -1)" ]; then   # old feddbe (kappa = 1) -> feddbe_k1
        for d in $dirs; do
            for f in "$d"/*; do
                g=$(basename "$f" | sed 's/\(^\|-\)feddbe\([-.]\|_summary\)/\1feddbe_k1\2/')
                [ -f "$f" ] && [ "$g" != "$(basename "$f")" ] && mv -n "$f" "$d/$g" && echo "$f -> $d/$g"
            done
        done
    fi
    for d in $dirs; do
        [ -d "$d" ] || continue
        for f in "$d"/*; do
            [ -f "$f" ] || continue
            g=$(basename "$f" | sed -e 's/-atmoe-/-fedsha-/' -e 's/FedATMoE/FedSHA/' \
                -e 's/\(^\|-\)gate_eq\([-.]\|_summary\)/\1fedsha\2/' -e 's/\(^\|-\)no_usage_proto\([-.]\)/\1no_gate\2/' \
                -e 's/\(^\|-\)feddbe_alt\([-.]\)/\1feddbe\2/')
            [ "$g" != "$(basename "$f")" ] && mv -n "$f" "$d/$g" && echo "$f -> $d/$g"
        done
    done
}

case "$1" in
    __args)      args_for "$2"; exit 0 ;;
    test_all)    g=${2:-$GPU}; shift 2 2>/dev/null || shift $#; loop_runs "$g" "$LOG_DIR/test_all.log" "--test_only 1" ${*:-$TEST_RUNS} ;;
    dump_all)    g=${2:-$GPU}; shift 2 2>/dev/null || shift $#; loop_runs "$g" "$LOG_DIR/dump_all.log" "--test_only 1 --dump_preds 1" ${*:-fedsha fedmenu} ;;
    migrate)     migrate ;;
    baselines)   run_group "$2" $BASELINES ;;
    personal)    run_group "$2" $PERSONAL ;;
    ablations)   run_group "$2" $ABLATIONS ;;
    list)        echo "baselines: $BASELINES"; echo "personal:  $PERSONAL"; echo "ablations: $ABLATIONS" ;;
    ""|-h|--help) sed -n '2,19p' "$0"; exit 1 ;;
    *)           run_one "$1" "${2:-$GPU}" "${3:-$1}" "${@:4}" ;;
esac
