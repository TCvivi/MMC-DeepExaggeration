# Magic-Mirror-Caricatures-Dataset-and-DeepExaggeration-system
The dataset and method for exaggerated portrait caricatures synthesis.
## Magic Mirror Caricatures Dataset
### Introduction
The **Magic Mirror Caricatures (MMC)** is a collection of portrait photos and exaggerated portrait caricatures derived from http://image.baidu.com/. The peculiarity of this database is that the characters in it are basically famous persons. With this interesting characteristic, for each exaggerated portrait caricature, a large number of portrait photos of the corresponding characters are available online. Thus, we construct the MMC by retrieving famous persons' portrait photos and exaggerated portrait caricatures from the Internet using exaggerated portrait caricatures keywords. Then, we can select the most matching pair of faces according to the angle and expression of the characters. The fina **547 pairs** of portrait photos and corresponding exaggerated portrait caricatures from **41** persons are collected. For each image, we use a points sequence including **68** labeled facial landmarks to present the geometric sturcture features. 
<div align=center> <img src="https://github.com/TCvivi/Magic-Mirror-Caricatures-Dataset-and-DeepExaggeration-system/blob/master/data.png" alt="data" /> </div>

### Download instructions
Please download the [dataset release agreement](https://github.com/TCvivi/Magic-Mirror-Caricatures-Dataset-and-DeepExaggeration-system/blob/master/Magic%20Mirror%20Caricatures%20Dataset%20Release%20Agreement.pdf), read it carefully and sign it. Notice, for students, the agreement should be signed by your supervisor. Please scan the signed document and send it to tangchenwei826@gmail.com. After the dataset application is approved, detailed download instructions will be sent back.

## Introduction of DeepExaggeration System
The DeepExaggeration is divided into training and testing stages. The training stage includes three steps, 1) transforming portrait photos and exaggerated portrait caricatures into points sequences for further geometric structural features extraction, 2) learning the injective mapping from geometric structures of portrait photos to those of exaggerated portrait caricatures by proposed Sequence-to-Sequence Variational AutoEncoder (Seq2Seq VAE) model, 3) pre-training the style transfer network by unpaired portrait photos and exaggerated portrait caricatures. The testing stage includes four steps, 1) transforming the test portrait photo into points sequence, 2) inputting points sequence into the Seq2Seq VAE model with fixed parameters and generating exaggerated points sequence, 3) deforming the test portrait photo according to the generated exaggerated points sequence, 4) adding the caricature style to the deformed portrait photo by the style transfer network with fixed parameters.
<div align=center> <img src="https://github.com/TCvivi/Magic-Mirror-Caricatures-Dataset-and-DeepExaggeration-system/blob/master/system.png" alt="system" /> </div>

### Contact
For questions or help, feel welcome to write an email to tangchenwei826@gmail.com
