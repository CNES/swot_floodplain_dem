Haditha testcase:

#
# PIXC

Tiles covering the area or part of it: 471_215R 178_094R

PIXC files are downloaded using the script: download_pixc_pixcvec.py (found in tools/)
It is located in the testcase directory.
Example of command to launch the code:
>> python download_pixc_pixcvec.py -d [your_path]/download_pixc_dir -prod SWOT_L2_HR_PIXC -pass 471 215 Right -c PIC0

For the code FPDEM, the PIXC files are stored in the directory: .[..]/Haditha/input/pixc (The path should be defined in the FPDEM script parameters file SWOT_Param_L2_HR_FPDEM_Haditha.rdf)

Some cycles seem to be bad or will cause issues for the FPDEM bathymetry extraction:
For tile 471_215R: 9, 13, 18, 25, 39

#
# Parameters file

The parameters file SWOT_Param_L2_HR_FPDEM_Haditha.rdf is located in the testcase directory.

#
# Start the code: slurm and env files

A slurm file fpdem.slurm is present in order to launch the code. 
>> sbatch fpdem.slurm

If not used, a python environment should be loaded first. One can be installed with the FPDEM_eodag_env.yml.
>> conda activate [your_path]/FPDEM_eodag_env
Then export the path to the src/ and scripts/ directories:
>> export PYTHONPATH=[your_path]/floodplain_dem/src/:$PYTHONPATH
>> export PYTHONPATH=[your_path]/floodplain_dem/scripts/:$PYTHONPATH

To launch the FPDEM script (complete [xxx] with the proper floodplain path):
>> python [xxx]/floodplain_dem/scripts/process_full_processing_floodplain.py [xxx]/floodplain_dem/testcases/Haditha/SWOT_Param_L2_HR_FPDEM_Haditha.rdf
