# Magic-Mirror-Caricatures-Dataset-and-DeepExaggeration-system
The dataset and method for exaggerated portrait caricatures synthesis.
## Introduction of Dataset
The **Magic Mirror Caricatures (MMC)** is a collection of portrait photos and exaggerated portrait caricatures derived from http://image.baidu.com/. The peculiarity of this database is that the characters in it are basically famous persons. With this interesting characteristic, for each exaggerated portrait caricature, a large number of portrait photos of the corresponding characters are available online. Thus, we construct the MMC by retrieving famous persons' portrait photos and exaggerated portrait caricatures from the Internet using exaggerated portrait caricatures keywords. Then, we can select the most matching pair of faces according to the angle and expression of the characters. The fina **547 pairs** of portrait photos and corresponding exaggerated portrait caricatures from **41** persons are collected.

**Examples of MMC**: (a) and (c) are the portrait photos. (b) and (d) are the corresponding exaggerated portrait caricatures.

![dataset](https://github.com/TCvivi/Magic-Mirror-Caricatures-Dataset-and-DeepExaggeration-system/blob/master/dataset.png)

For each image, we use a points sequence including **68** labeled facial landmarks to present the geometric sturcture features. 

![landmark](https://github.com/TCvivi/Magic-Mirror-Caricatures-Dataset-and-DeepExaggeration-system/blob/master/sample.png)


