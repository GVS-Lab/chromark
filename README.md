# Single-cell imaging-AI based chromatin biomarkers for proton therapy efficacy in tumor patients using liquid biopsies

The repository contains the code used to run the analyses presented in our publication:

> [**Single-cell imaging-AI based chromatin biomarkers for proton therapy efficacy in tumor patients using liquid biopsies (Under review)**]()

<p align="center" width="100%">
  <b>Chromatin organization of PBMCs reflects the presence of tumor signals</b> <br>
    <img width="66%" src="https://github.com/GVS-Lab/immune_cell_project/blob/1dae3aa5ca72b08fd3c3abaad73b43b76cc3adc6/chromark_paper_overview.png">
</p>

----

# System requirements

The code has been developed and executed on a HP Z4 workstation running Ubuntu 20.04.5 LTS with a Intel(R)
Xeon(R) W-2255 CPU with 3.70 GHz, 128GB RAM. Note that the code can also be run for machines with less available RAM.

## Installation

To install the code, please clone the repository and install the required software libraries and packages listed in
the **requirements.txt** file:

```
git clone https://github.com/GVS-Lab/immune_cell_project.git
conda create --name icp --file requirements.txt
conda activate icp
```

## Data resources
The data to rerun the results can be obtained from the authors upon request and will be made publicly available upon publication.

---

# Reproducing the paper results

## 1. Data preprocessing (Optional)
The full preprocessing pipeline including the image segmentation in 2D and 3D as well as the feature extraction and (if applicable) the cell type classification can be run via the following command.
```
python run.py --config config/full_pipeline/config_pbmc_full_marker.yml
```
Note that the config file needs to be adjusted such that the correct file locations are specified.
While all other parameters are recommended to be kept the channel parameters should also be adjusted to reflect the channels present in the images that are to be preprocessed.

*The above steps can be skipped, if the preprocessed data is used which can be obtained from the authors (see Data resources)*

## 2. Reproducing the figures' results

All results presented in the main and supplemental figures can be obtained by running the respective jupyter notebooks located in ``jupyter/notebooks/figures``. To run those notebooks, please start the jupyter server via
```
jupyter notebook
```
navigate to the notebook and run all cells.


---

# How to cite

If you use any of the code or resources provided here please make sure to reference the required software libraries if
needed and also cite our work:

**TO BE ADDED**
---
