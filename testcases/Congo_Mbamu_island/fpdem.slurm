#!/bin/bash
#SBATCH --job-name=FPDEM_Congo
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --account=swotce
#SBATCH --output=/work/SWOT/swotdev/swotdev/lemrabl/floodplaindem/testcases/Congo_Mbamu_island/output/FPDEM_Congo.out
#SBATCH --error=/work/SWOT/swotdev/swotdev/lemrabl/floodplaindem/testcases/Congo_Mbamu_island/output/FPDEM_Congo.err
#SBATCH --export=none

# Load the conda environment
module load conda
conda activate /work/scratch/env/lemrabl/.conda/envs/FPDEM_eodag_env_final
export PYTHONPATH=/work/SWOT/swotdev/swotdev/lemrabl/floodplaindem/algos/floodplain_dem/src/:$PYTHONPATH
export PYTHONPATH=/work/SWOT/swotdev/swotdev/lemrabl/floodplaindem/algos/floodplain_dem/scripts/:$PYTHONPATH

# Export certificate to access internet during the run (needed to load basemaps with cartopy)
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_DIR=/etc/pki/tls/certs

# Launch the FPDEM algorithm
python /work/SWOT/swotdev/swotdev/lemrabl/floodplaindem/algos/floodplain_dem/scripts/process_full_processing_floodplain.py /work/SWOT/swotdev/swotdev/lemrabl/floodplaindem/algos/floodplain_dem/testcases/Congo_Mbamu_island/SWOT_Param_L2_HR_FPDEM_Congo.rdf
