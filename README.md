# Cattle Counting in Large Pasture Areas

This repository contains the code and datasets for the paper "Multi-attribute, graph-based approach for duplicate cattle removal and counting in large pasture areas from multiple aerial images". The paper is available for free [here](https://authors.elsevier.com/a/1ioa%7EcFCSbG6N).

## Overview

The paper addresses the challenges of counting cattle in large pasture areas using UAVs equipped with high-resolution cameras. It proposes a novel graph-based method that incorporates multi-attributes such as velocity, direction, state, color, and distance to improve duplicate removal and counting accuracy.


## Code

The code provided in this repository is for cattle counting tasks. It takes as input a folder containing images taken by a drone orthogonal to the ground and XML files (one for each image and same name) containing the bounding box information for each cattle present in the image. In case of manual labeling, we recommend using the app [LabelImg](https://github.com/HumanSignal/labelImg) to identify the cattle in all images that will be processed and counted.

### Python files of Cattle Counting Method

- `setup.sh`: A file that, when executed, creates a virtualenv and installs all Python libraries needed to make our counting method work.
- `requirement`: A file that lists all libraries and versions needed to download and install.
- `countingMethod.py`: The main class, which iterates through all images of the defined folder, performs the cattle counting, and returns the total.
- `computeEdges.py`: Responsible for creating weighted edges based on cattle attributes, computing the probability of two cattle being the same.
- `config.py`: Contains parameters, features, and general configuration used in the counting method.
- `logisticRegression.py`: Responsible for training and predicting the probability of two cattle being the same based on velocity.
- `maximumFlow.py`: Implements a weighted version of the Ford-Fulkerson algorithm to detect duplicate cattle.
- `projection.py`: Responsible for computing the coordinates of the projection of the image (corners of the image).

#### Folders
- `models`: Contains the models to predict color, state, direction, and velocity.
- `playground`: Contains images examples to test the counting method.

### Installation and Execution

To install, execute the `setup.sh` script. To execute, use the command `python3 countingMethod.py`.

The `projection.py` file can be executed standalone to compute the projections of some image.


## Datasets

Datasets public available:

- [Whole Training dataset](https://drive.google.com/drive/folders/1tb4COoj1w7bEuW8hKJ5HBsCccOvXW0gl?usp=sharing): Training dataset for cattle detection and classification.
- [BR_set](https://drive.google.com/drive/folders/1Z65UhIOEWdpubdM0XjBm4L0DfICIj6hF?usp=drive_link): Test collection containing photos from many farms in Brazil.
- [T2606](https://drive.google.com/drive/folders/1Z65UhIOEWdpubdM0XjBm4L0DfICIj6hF?usp=drive_link): comprises a total of 2606 individual cattle images, cropped from aerial photographs taken in 2017, 2018, and 2020 at several farms in Brazil.
- [Pasto4-16-10-7h](https://drive.google.com/drive/folders/1ceEACmSwR2FWcqGNfXp-p7NB8mP2Gu1L?usp=drive_link): Obtained in the Pasture 4 in Água Boa farm at 16/10/2020 7:00 am.
- [Sede-18-10-7h](https://drive.google.com/drive/folders/1Wq8ZiHgnfYhhVMyBwq_7deeioJMZDmst?usp=drive_link): Obtained in the Sede pasture in Água Boa farm at 18/10/2020 7:00 am.
- [Brejo-19-10-12h](https://drive.google.com/drive/folders/1mpAxrD7_0UAkH2W23dA0rS1t24gzPpVK?usp=drive_link): Obtained in the Brejo pasture in Água Boa farm at 19/10/2020 12:00 pm.
- [Pasto2-17-10-7h](https://drive.google.com/drive/folders/1CYdyGYtXkeS5Z9vNnjuXNumfxX9kXo-0?usp=drive_link): Obtained in the Pasture 2 in Água Boa farm at 17/10/2020 7:00 am.
- [Pasto1-15-10-12h](https://drive.google.com/drive/folders/19jgEfaYYyO7WtC6n_yX8bFjLNN9SYLM1?usp=drive_link): Obtained in the Pasture 1 in Água Boa farm at 15/10/2020 12:00 pm.

## Citation

If you use this code or the datasets in your research, please cite our paper:

Soares, V.H.A., Ponti, M.A., & Campello, R.J.G.B. (2024). Multi-attribute, graph-based approach for duplicate cattle removal and counting in large pasture areas from multiple aerial images. Computers and Electronics in Agriculture, 220, 108828. DOI: https://doi.org/10.1016/j.compag.2024.108828

#### BibTeX:

```
@article{SOARES2024108828,
title = {Multi-attribute, graph-based approach for duplicate cattle removal and counting in large pasture areas from multiple aerial images},
journal = {Computers and Electronics in Agriculture},
volume = {220},
pages = {108828},
year = {2024},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2024.108828},
url = {https://www.sciencedirect.com/science/article/pii/S0168169924002199},
author = {V.H.A. Soares and M.A. Ponti and R.J.G.B. Campello},
keywords = {Cattle counting, Duplicate removal, Precision farming, Unmanned Aerial Vehicles (UAVs), Graph-based method, Livestock},
abstract = {In this study, we address the challenges of counting cattle in large pasture areas using Unmanned Aerial Vehicles (UAVs) equipped with high-resolution cameras. Traditional manual counting methods are laborious and error-prone, while existing automated approaches struggle with duplicate animal detection. To overcome these limitations, we propose a novel graph-based method that incorporates multiple-attributes, including velocity, direction, state (lying down or standing), color, and distance, to improve duplicate removal and counting accuracy. We conducted extensive experiments involving automated hyper-parameter learning to effectively integrate these attributes into our method. By employing a Ford–Fulkerson graph algorithm, we detect and remove duplicated cattle based on their multiple attributes. An ablation study validates the contribution of each attribute. Additionally, we provide new datasets of drone images captured in large pastures to support research in this field. Our results demonstrate that our proposed method outperforms state-of-the-art techniques, achieving an average percentage error of 2.34%. Comparisons with mosaic-based and other graph-based methods validate the effectiveness of our approach, which contributes to more efficient cattle counting practices and enhances livestock management in agriculture.}
}
```
