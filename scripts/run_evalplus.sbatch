#!/bin/bash
#SBATCH --job-name=evalplus
#SBATCH --partition=def
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
exec bash /scratch/project/prj-02-llm-reasoning-kalathil/vishnukunde/coding-agent-rl/scripts/run_evalplus.sh
