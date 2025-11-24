#!/bin/bash
# run_eval.sh: Run the evaluation pipeline for Brain-to-Text
# Usage: ./run_eval.sh [--max_trials N] [--eval_type TYPE]
#   --max_trials N   Limit evaluation to N trials (smoke test)
#   --eval_type TYPE Set evaluation type (default: test)

EVAL_ENV=b2txt25_lm
CONDA_ENV=b2txt25_lm
EVAL_TYPE="test"
MAX_TRIALS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --max_trials)
        MAX_TRIALS="$2"
        shift; shift
        ;;
        --eval_type)
        EVAL_TYPE="$2"
        shift; shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

CMD="bash scripts/run_with_services.sh -- --eval_type $EVAL_TYPE"
if [[ -n "$MAX_TRIALS" ]]; then
    CMD+=" --max_trials $MAX_TRIALS"
fi

# Run the evaluation pipeline
EVAL_ENV=$EVAL_ENV conda run -n $CONDA_ENV $CMD
