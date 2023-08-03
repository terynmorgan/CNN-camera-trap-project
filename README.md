# Deep Learning Image Classification of Ecological Camera Traps
Project done through INFO-B 529: Machine Learning for Bioiformatics. 

**Spring 2022** <br/>
**Programming Language:** Python <br/>
**Background:** <br/>
Analyzing data from ecological camera traps can enable a better understanding of realtionships within ecosymstems, population density estimation, impacts of invasive species, and biological interactions over time. However, false triggers from non-animal movement and the high levels of noise contained in these images greatly increase the volume of data produced. 

Since information extraction from images is typically done through manual, human effort, this project aimed to develop a Convolutional Nueral Network that accepts ecological camera trap images and a metadata csv file as input to distinguish between animal images and noise. 

**Data:** <br/>
Wellington Camera Traps from the Labeled Information Library of Alexandria: Biology and Conservation (LILA BC): https://lila.science/datasets/wellingtoncameratraps
Data set contains 270,450 images from 187 locations in Wellington, New Zealand. Images are classified into 1 of 15 different distinct animal categories with 17% of images being labled as empty or unclassifiable. 

**Files:** <br/>
wellington_camera_traps.csv -> Csv file containing labels for camera trap images
download_data.py -> Python file used to download images from LILA BC
CNN.py -> Python file used for CNN architecture construction

**Results:** <br/>
After training the CNN model for 16 epochs using a batch size of 32, the model achieved an accuracy of 90.45% and loss of 0.3055 on the training dataset. However, on the testing data set, the model  achieved an accuracy of 80.10% and loss of 1.1591. 
