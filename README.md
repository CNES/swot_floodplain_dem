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

The FPDEM algorithm can be obtained on the CNES github: https://github.com/CNES/swot_floodplain_dem

Use the command git clone.

## Environment creation

Within the FPDEM repository, the file FPDEM_eodag_env.yml can be used to set up the FPDEM conda environment. 

```
$ conda env create -f FPDEM_eodag_env.yaml [-p <path_to_environment>] 
```

This environment must be activated before launching the FPDEM algorithm.  

## PIXC download

First, the file eodag.yml (to be placed in : ~/.config/eodag/eodag.yml) needs to be set up to use the PIXC downloading scripts. 
The providers 'swot' and 'hydroweb_next' are the providers used in the code.
Edit the EODAG configuration file and insert the appropriate authentication information using your personal SWOT and/or Hydroweb.next credentials. 

To complete the swot provider configuration, you will need:
    - A REGARDS account (email and password)
    - Set up the EMAIl and PASSWORD in eodag.yml

To complete the hydroweb_next provider configuration, you will need:
  - A Hydroweb.next account (username and password), which can be obtained by registering on the Hydroweb.next portal here: https://hydroweb.next.theia-land.fr/
  - Set up a Hydroweb.next API key, which can be generated from your Hydroweb.next user account.

To download the PIXC products covering the AOI, several scripts are available in the `tools/` directory:

- **Downloading from the TREX cluster**:
  - `download_SWOT_products.py`
  - `download_pixc_pixcvec.py`

- **Downloading from a local machine**:
  - `download_SWOT_products.py`
  - `download_pixc_pixcvec_podaac.py`

In order to know the different parameters available to select the PIXC tiles, use: 

```
$ python download_pixc_pixcvec.py -h
```

An example of a command to launch the code is: 

```
$ python download_pixc_pixcvec.py -d [downloading_directory] -prov hydroweb_next -prod SWOT_L2_HR_PIXC SWOT_L2_HR_PIXCVEC -pass 264 69 Right -c PIC0
```

where -d is the argument to choose the downloading directory, -prov the provider, -prod the wanted products (here PIXC and PIXCVec), -pass the pass 269, tile 69 and tile side Right, -c the CRID PIC0.

This script will start by looking onto the providers 'swot' and 'hydroweb_next', if none is specified, to find the list of products corresponding to the user arguments. 
It will show the list and ask the user to continue to the downloading part. 

## Running the FPDEM algorithm

The first step to run the FPDEM code is to go into one of the testcases directory located in /floodplain/run/. 
Several testcases are available: 

- Orient (lake, France)
- Barotse (floodplain / river, Zambia)
- Lajeodo (river, Brazil)
- Bijagos (coastal archipelago, Guinea-Bissau)
- Congo_Mbamu_island (river island, Democratic Republic of the Congo)
- Haditha (river / reservoir, Iraq)
- Koshi (braided river / floodplain, Nepal)
- Tele_Mali (floodplain wetland, Mali)
- Toshka (desert lakes, Egypt)
- Wadden (tidal flats / coastal wetland, Netherlands-Germany-Denmark)

Four files are present in each testcase directory:
- A readme file
- The parameter file : SWOT_Param_L2_HR_FPDEM_XXX.rdf
- The slurm file : fpdem.slurm
- A notebook file : FPDEM_XXX_testcase.ipynb

The FPDEM algorithm can be executed on a computing cluster using the SLURM scheduler, locally from a terminal or a Python IDE, or from a Jupyter Notebook environment.

### Notebook execution

The FPDEM workflow can be explored and executed through Jupyter notebook. 
Before starting a notebook, make sure that the conda environment is activated.
If required, additional certificates should also be exported to enable the display of basemaps in Cartopy figures.

Launch Jupyter Notebook from the project directory:

```
$ jupyter notebook
```

or launch JupyterLab:

```
$ jupyter lab
```

Then open the desired notebook from your browser and select the appropriate conda kernel.

### Cluster execution (SLURM)

To launch the processing on a SLURM-managed cluster:

```
$ sbatch fpdem.slurm
```

The `fpdem.slurm` script automatically activates the conda environment and performs the required exports. 
Ensure that all paths defined in the SLURM script are correctly configured before submission.

### Local execution

For local execution, make sure that the conda environment is activated.
After activating the conda environment, the `PYTHONPATH` must be configured to include the `src/` and `scripts/` directories:

```
$ export PYTHONPATH=[your_path]/floodplain_dem/src/:$PYTHONPATH
$ export PYTHONPATH=[your_path]/floodplain_dem/scripts/:$PYTHONPATH
```

Certificates should also be exported to enable the display of basemaps in Cartopy figures.

The complete processing chain can be launched with:

```
$ python ../../scripts/process_full_processing_floodplain.py SWOT_Param_L2_HR_FPDEM_Barotse.rdf
```

Alternatively, each processing step can be run independently:

```
$ python ../../scripts/process_floodplain.py SWOT_Param_L2_HR_FPDEM_Barotse.rdf

$ python ../../scripts/process_extract_area.py SWOT_Param_L2_HR_FPDEM_Barotse.rdf

$ python ../../scripts/process_raster.py SWOT_Param_L2_HR_FPDEM_Barotse.rdf
```

Running the workflow step by step can be useful for reprocessing only a specific stage of the pipeline.


<span style="display:none;">
## Liens utiles
- [Confluence de l'Usine Logicielle](https://confluence.cnes.fr/pages/viewpage.action?pageId=17961975)
    - [Manuel pour l'utilisation de GitLab](https://confluence.cnes.fr/display/USINELOG/GitLab+-+Manuel+utilisateur)
    - [Comment choisir son workflow Git](https://confluence.cnes.fr/display/USINELOG/Gestion+de+configuration+-+Choix+du+flow+Git)
    - [Manuel pour l'utilisation de GitLab-CI](https://confluence.cnes.fr/display/USINELOG/GitLab-CI)
- [Demandes de support à l'UL](https://confluence.cnes.fr/display/USINELOG/Les+demandes+de+support)
- Vos propres pages de documentation :D
</span>
