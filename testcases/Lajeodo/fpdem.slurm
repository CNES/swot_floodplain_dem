#!/bin/bash
#SBATCH --job-name=FPDEM_Lajeodo
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --account=swotce
#SBATCH --export=none

# Load the conda environment
source /work/SWOT/swotdev/swotdev/stephag/env_FPDEM_eodag.sh

# Export certificate to access internet during the run (needed to load basemaps with cartopy) 
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_DIR=/etc/pki/tls/certs

# Launch the FPDEM algorithm
python /work/SWOT/swotdev/swotdev/stephag/floodplain_dem/scripts/process_full_processing_floodplain.py /work/SWOT/swotdev/swotdev/stephag/floodplain_dem/testcases/Lajeodo/SWOT_Param_L2_HR_FPDEM_Lajeodo.rdf

