# Adaptive Multisensor Acquisition via Spatial Contextual Information for Compressive Spectral Image Classification

This repository contain the code to reproduce the results presented the paper following paper:

*Diaz, Nelson, et al. "Adaptive Multisensor Acquisition via Spatial Contextual Information for Compressive Spectral Image Classification", IEEE JSTART, 2021.



This repository aims to adaptive design the coded apertures from compressive spectral images acquired with a multisensor camera. The classification is performed using the captured compressive projections.

## Usage
Download the CompressiveAdaptiveClassificationMultisensor repository
1. Download this repository via git 
```
git clone https://github.com/nelson10/CompressiveAdaptiveClassificationMultisensor.git
```
2. To run the code using either the function Main3DCASSI or MainCCASSI that perform the sensing, Adaptive coded aperture design and classification using the compressive measurements.


## Datasets

The datasets could be download from the following [link](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

The files should be placed in the Data folder of this repository.

If you use this code or dataset please consider citing our paper with the following Bibtex code:

```
@article{Diaz2021,
author = "Nelson Diaz and Juan Marcos and Esteban Vera and Henry Arguello",
title = "{Adaptive Multisensor Acquisition via Spatial Contextual Information for Compressive Spectral Image Classification}",
year = "2021",
month = "7",
url = "https://www.techrxiv.org/articles/preprint/Adaptive_Multisensor_Acquisition_via_Spatial_Contextual_Information_for_Compressive_Spectral_Image_Classification/14988804",
doi = "10.36227/techrxiv.14988804.v1"
}
```

A pre-print is available at this [link](https://www.techrxiv.org/articles/preprint/Adaptive_Multisensor_Acquisition_via_Spatial_Contextual_Information_for_Compressive_Spectral_Image_Classification/14988804)

## Abstract

Spectral image classification uses the huge amount of information provided by spectral images to identify objects in the scene of interest. In this sense, spectral images typically contain redundant information that is removed in later processing stages. To overcome this drawback, compressive spectral imaging (CSI) has emerged as an alternative acquisition approach that captures the relevant information using a reduced number of measurements. Various methods that classify spectral images from compressive projections have been recently reported whose measurements are captured by non-adaptive, or adaptive schemes discarding any contextual information that may help to reduce the number of captured projections. In this paper, an adaptive compressive acquisition method for spectral image classification is proposed. In particular, we adaptively design coded aperture patterns for a dual-arm CSI acquisition architecture, where the first system obtains compressive multispectral projections and the second arm registers compressive hyperspectral snapshots. The proposed approach exploits the spatial contextual information captured by the multispectral arm to design the coding patterns such that subsequent snapshots acquire the scene's complementary information improving the classification performance. Results of extensive simulations are shown for two state-of-the-art databases: Pavia University and Indian Pines. Furthermore, an experimental setup that performs the adaptive sensing was built to test the performance of the proposed approach on a real data set. The proposed approach exhibits superior performance with respect to other methods that classify spectral images from compressive measurements.
