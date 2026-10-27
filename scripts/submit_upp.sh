#!/usr/bin/env bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 32000
#SBATCH --partition private-dpnc-cpu
#SBATCH --time 2-00:00:00

# UPP preprocessing with configs/upp/fcc-zh.yaml. Extra args are forwarded to
# upp_commands.sh, e.g.
#   sbatch scripts/submit_upp.sh
#   sbatch scripts/submit_upp.sh merge train

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
COMMANDFILE="/repo/scripts/upp_commands.sh"

# Define the actual command
COMMAND=(
	srun apptainer exec
	--contain
	--pwd /repo
	${FCC_BINDS}
	${FCC_ENVFLAGS}
	"${FCC_IMAGE}"
	"${COMMANDFILE}"
)

# Forward all user-provided args (0, 1, 2, ... n)
"${COMMAND[@]}" "$@"
