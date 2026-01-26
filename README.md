# FloodplainDEM
The SWOT project FloodPlainDEM (or FPDEM) aims to determine and extract the dry bathymetry on different Area of Interest (AOI) like flood plains areas, rivers, lakes and even estuaries.  

This document explains how to recover the FPDEM repository, create the conda environment, download PIXCs and launch the FPDEM code.


- [Contexte](#contexte)
- [Code download](#installation)
- [Environment creation](#installation)
- [PIXC download](#utilisation) 
- [Code execution](...)
<!-- - [Liens utiles](#liens-utiles) -->

## Contexte

Contacts :
- Leader CNES : Damien Desroches
- Other : Gwendoline Stéphan

The FPDEM algorithm contains 3 main scripts and 1 workflow script (which launches the 3 main scripts one after the other).
- Workflow script : process_full_processing_floodplain.py
- Main scripts: process_floodplain.py, process_extract_area.py, process_raster.py (run in this order by the workflow script)


## Code download

The code FPDEM can be obtained on the CNES gitlab: https://gitlab.cnes.fr/desrochesd/floodplain_dem#utilisation

Use the command git clone. 

Note: The access needs to be configured beforehand in order to clone the floodplain repository. 

## Environment creation

Within the FPDEM repository, the file FPDEM_eodag_env.yaml can be used to set up the FPDEM conda environment. 

```
$ conda env create -f FPDEM_eodag_env.yaml [-p <path_to_environment>] 
```

This environment must be activated before launching the code FPDEM.  

## PIXC download

First, the file ~/.config/eodag/eodag.yml needs to be set up to use the PIXC downloading script. The providers 'swot' and/or 'hydroweb_next' are the providers used in the code. For the former, the user needs to have an account on REGARDS. For the latter, access to hydroweb_next website is required, and an apikey is needed. 

To download the pixel cloud products of the AOI, the user can use the code download_pixc_pixcvec.py. This code can be found inside the testcases directories in floodplain/run/testcase_XXX. 

The downloading script uses Eodag. No provider is set up in the script, so it tries to find the products on the different providers if available. 

In order to know the different parameters available to select the PIXC tiles, the following command can be used: 

```
$ python download_pixc_pixcvec.py -h
```

An example of a command can as follows: 

```
$ python download_pixc_pixcvec.py -d [downloading_directory] -prov hydroweb_next -prod SWOT_L2_HR_PIXC SWOT_L2_HR_PIXCVEC -pass 264 69 Right -c PIC0 -conv 1
```

where -d is the argument to choose the downloading directory, -pd the wanted products (here PIXC and PIXCVec), -p the pass 269, tile 69 and tile side Right, -c the CRID PIC0 and -cv the flag to choose if the user wants to convert the netcdf product into shapefile. 

The script will start by looking onto the different providers to find a list of products corresponding to the user arguments set up. It will show the list and ask the user to continue to the downloading part if the answer is yes. 

## Code execution

The first step to run the FPDEM code is to go into one of the testcases directory located in /floodplain/run/. Several testcases are available: 

- Orient (lake, France)
- Barotse (floodplain/river, Zambia)
- Lajeodo (river, Brazil)

Four files are present in each testcase directory:
- A readme file
- The parameter file : SWOT_Param_L2_HR_FPDEM_XXX.rdf
- The slurm file : fpdem.slurm
- A notebook file : FPDEM_XXX_testcase.ipynb


To launch the FPDEM code with slurm use the following command:
```
$ sbatch fpdem.slurm
```

To launch the code without slurm :
```
$ python ../../scripts/process_full_processing_floodplain.py SWOT_Param_L2_HR_FPDEM_Barotse.rdf
```
or 
```
$ python ../../scripts/process_floodplain.py SWOT_Param_L2_HR_FPDEM_Barotse.rdf
$ python ../../scripts/process_extract_area.py SWOT_Param_L2_HR_FPDEM_Barotse.rdf
$ python ../../scripts/process_raster.py SWOT_Param_L2_HR_FPDEM_Barotse.rdf
```




<span style="display:none;">
## Liens utiles
- [Confluence de l'Usine Logicielle](https://confluence.cnes.fr/pages/viewpage.action?pageId=17961975)
    - [Manuel pour l'utilisation de GitLab](https://confluence.cnes.fr/display/USINELOG/GitLab+-+Manuel+utilisateur)
    - [Comment choisir son workflow Git](https://confluence.cnes.fr/display/USINELOG/Gestion+de+configuration+-+Choix+du+flow+Git)
    - [Manuel pour l'utilisation de GitLab-CI](https://confluence.cnes.fr/display/USINELOG/GitLab-CI)
- [Demandes de support à l'UL](https://confluence.cnes.fr/display/USINELOG/Les+demandes+de+support)
- Vos propres pages de documentation :D
</span>

