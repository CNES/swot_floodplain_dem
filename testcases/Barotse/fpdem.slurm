#!/bin/bash
#SBATCH --job-name=FPDEM_Barotse
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --account=swotce
#SBATCH --export=none
 
source /work/SWOT/swotdev/swotdev/stephag/env_FPDEM_eodag.sh

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_DIR=/etc/pki/tls/certs

python /work/SWOT/swotdev/swotdev/stephag/floodplain_dem/scripts/process_full_processing_floodplain.py /work/SWOT/swotdev/swotdev/stephag/floodplain_dem/testcases/Barotse/SWOT_Param_L2_HR_FPDEM_Barotse.rdf
