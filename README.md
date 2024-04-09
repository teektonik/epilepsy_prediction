# Predicting epileptic seizures
**Authors**:
* [Mikhail Komarov](https://github.com/AsphodelRem) `leader` `dev`
* [Ksenia Belkova](https://github.com/didilovu) `scrum master`
* [Fedor Tikunov](https://github.com/FedorTikunov) `dev`
* [Antonina Testova](https://github.com/teektonik) `QA`
## Problem
Each year, about 5 million people worldwide are diagnosed with epilepsy. Epilepsy is a brain disorder that causes recurrent seizures. Epilepsy usually causes seizures, but some people have other symptoms, such as confusion and staring off into space, etc.

## Decision
As a rule, epileptic seizures come unexpectedly, which can be fatal. Our goal is to develop an ML system that will predict a seizure before it starts. 
This will give a person the opportunity to take control of the situation and take the necessary medication to block the seizure.

In order for the detector to work correctly, the sensors must read the following signals from the patient: accelerometer, BVP, EDA. The device with the model is triggered 1.5 minutes before the onset of a seizure. To create such an ML system, we needed to properly process the data and train a model on that data.

## Data
We trained the model on the [Seizure Gauge data](https://www.epilepsyecosystem.org/) dataset by Dr. Levin Kuhlman. 

We were provided with patient information. We processed this data, deleted corrupted files, removed noise, and converted it to a segmented view.

For more accurate detection, we did not have enough segments where an epileptic seizure was present, so we decided to augment these segments. 
In this way, we got rid of the class imbalance and improved our metrics.

The data were fed into the model in the matrix form. The description of the model will be presented below. 

## Model

![model (2)](https://github.com/teektonik/epilepsy_prediction/assets/124969658/b55a39bb-ee49-4ab2-9505-ab83a81b773c)

## Result
At the moment our metrics are as follows: 
- precision - 92%, 
- recall - 96%, 
- F1 - 94%. 

The detector sees epilepsy approaching 1.5 minutes before it.
