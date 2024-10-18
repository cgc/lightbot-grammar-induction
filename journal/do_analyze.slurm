#!/bin/bash
#SBATCH --job-name=lightbot-model-fitting
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=25G
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cgcorrea@princeton.edu

PROJ=/mnt/cup/people/cgcorrea/pu/lightbot-grammar-induction

cd $PROJ
. env/bin/activate
python -m journal.do_analyze --nsampled=50 --workers=12 --analysis=$1
