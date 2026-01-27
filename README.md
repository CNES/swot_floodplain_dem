# FloodplainDEM
The SWOT project FloodPlainDEM (or FPDEM) aims to determine and extract the dry bathymetry on different Area of Interest (AOI) like flood plains, rivers, lakes and even estuaries.  

This document explains how to recover the FPDEM repository, create the conda environment, download PIXCs and launch the FPDEM algorithm.


- [Contexte](#contexte)
- [Code download](#code-download)
- [Environment creation](#environment-creation)
- [PIXC download](#pixc-download) 
- [Code execution](#code-execution)
<!-- - [Liens utiles](#liens-utiles) -->

## Contexte

Contacts :
- Leader CNES : Damien Desroches
- CS Group : Gwendoline Stéphan

The FPDEM algorithm contains 3 main scripts and 1 workflow script (which launches the 3 main scripts one after the other).
- Workflow script : process_full_processing_floodplain.py
- Main scripts: process_floodplain.py, process_extract_area.py, process_raster.py (launched in this order by the workflow script)


## Code download

The FPDEM algorithm can be obtained on the CNES gitlab: https://gitlab.cnes.fr/desrochesd/floodplain_dem#utilisation

Use the command git clone. 

NB: The access needs to be configured beforehand in order to clone the floodplain repository. 

## Environment creation

Within the FPDEM repository, the file FPDEM_eodag_env.yml can be used to set up the FPDEM conda environment. 

```
$ conda env create -f FPDEM_eodag_env.yaml [-p <path_to_environment>] 
```

This environment must be activated before launching the FPDEM algorithm.  

## PIXC download

First, the file eodag.yml (to be placed in : ~/.config/eodag/eodag.yml) needs to be set up to use the PIXC downloading script. The providers 'swot' and 'hydroweb_next' are the providers used in the code. For 'swot', the user needs to have an account on REGARDS and set up the EMAIl and PASSWORD in eodag.yml. For 'hydroweb_next', access to hydroweb_next website is required, and an apikey is needed and needs to be set up in eodag.yml. 

To download the PIXC products encompassing the AOI, the user can use the code download_pixc_pixcvec.py located in tools/.

In order to know the different parameters available to select the PIXC tiles, the following command can be used: 

```
$ python download_pixc_pixcvec.py -h
```

An example of a command to launch the code is: 

```
$ python download_pixc_pixcvec.py -d [downloading_directory] -prov hydroweb_next -prod SWOT_L2_HR_PIXC SWOT_L2_HR_PIXCVEC -pass 264 69 Right -c PIC0 -conv 1
```

where -d is the argument to choose the downloading directory, -prov the provider, -prod the wanted products (here PIXC and PIXCVec), -pass the pass 269, tile 69 and tile side Right, -c the CRID PIC0 and -conv the flag to choose if the user wants to convert the netcdf product into shapefile. 

The script will start by looking onto the providers 'swot' and 'hydroweb_next', if none is specified, to find the list of products corresponding to the user arguments. It will show the list and ask the user to continue to the downloading part. 

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

The first step after loading the conda environment is to export the PYTHONPATH for the scripts/ and src/ directories:
```
$ export PYTHONPATH=[your_path]/floodplain_dem/src/:$PYTHONPATH
$ export PYTHONPATH=[your_path]/floodplain_dem/scripts/:$PYTHONPATH
```

To launch the FPDEM code with slurm use the following command (the loading of the environment and the export mentionned above are performed within the fpdem.slurm so make sure to define the paths corectly):
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

