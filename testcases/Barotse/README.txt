Barotse testcase:

#
# PIXC

Tiles covering the area or part of it: 305_128R  208_181L

PIXC files are downloaded using the script: download_pixc_pixcvec.py 
It is located in the testcase directory.
Example of command to launch the code:
>> python download_pixc_pixcvec.py -d [xxx]/download_pixc_dir -prod SWOT_L2_HR_PIXC -pass 305 128 Right -c PIC0 -conv 1

For the code FPDEM, the PIXC files are stored in the directory: testcase_Barotse/input/pixc (The path can be defined in the FPDEM script parameters file SWOT_Param_L2_HR_FPDEM_20250324T000000_Barotse.rdf)

#
# Parameters file

The parameters file SWOT_Param_L2_HR_FPDEM_20250324T000000_Barotse.rdf is located in the testcase directory.

#
# Start the code: slurm and env files

A slurm file fpdem.slurm is present in order to launch the code. 
>> sbatch fpdem.slurm

If not used, a python environment should be loaded first.
One can be found in /work/scratch/env/stephag/.conda/envs/FPDEM_env
>> conda activate /work/scratch/env/stephag/.conda/envs/FPDEM_env
Otherwise you can source the file env_GS_new.sh located in floodplain/. The python paths necessary for the code will also be loaded then.

To launch the FPDEM script (complete [xxx] with the proper floodplain path):
>> python [xxx]/floodplain/scripts/process_full_processing_floodplain.py [xxx]/floodplain/run/testcase_Barotse/SWOT_Param_L2_HR_FPDEM_20250324T000000_Barotse.rdf
