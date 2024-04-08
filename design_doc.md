# Predicting epileptic seizures


> ## Terms & Explanations
> - EDA - electrodermal activity, refers to changes in the electrical conductivity of the skin in response to sweat secretion. EDA is related to the regulation of our internal temperature and to emotional arousal.
> - BVP - pulse blood volume. The amount of blood flowing through a given segment of artery during each pulse period.
> - HRV - heart rate variability. This is an index that reflects the irregularity of a person's heartbeat.
> - Accelerometer - an acceleration measurement device that acts as a sensor for changes in the device's position in space.
> - SQI - Signal Quality Index, which displays the quality of the received signal in a coefficient from 0 to 100, where 0 - very poor signal quality, i.e. there is practically no signal and 100 - maximum quality signal.




## 1. Objectives & Prerequisites
### 1.1. Why go into ML system development?  

- The development of this product will help not only people with epilepsy, but also their loved ones and those around them. 
- We will consider the development a success if the accuracy of the prediction reaches 90% percent.
- This product can be used by anyone who wants to use it. 

### 1.2. Decision prerequisites

The patient must be monitored in daily life, otherwise there is a risk that an attack will take him or her by accident. 

Based on these considerations there is a need for a product capable of predicting a seizure and enabling the timely use of the necessary medication to block an epileptic seizure.

## 2. Methodology    

### 2.1. Problem statement 

Our goal is to develop an ML system that can notify a seizure well in advance of its onset. 

The difficulty of the task is in detecting anomalies. A person's heart rate accelerates significantly during physical activity or emotional arousal. We need to teach the model not to react in such situations, but at the same time not to miss the approach of an epileptic seizure.

### 2.2 Stages of problem solving
> Stages:
> - Stage 1 - obtaining the dataset
> - Stage 2 - data preparation
> - Stage 3 - preparation of predictive models
> - Stage 4 - interpretation of models

*Stage 1*
  
| Name of data  | Source | Resource required to retrieve data | Whether data quality has been verified|
| ------------- | ------------- | ------------- | ------------- |
| Seizure Gauge data | Dr. Levin Kuhlman|  | + |

*Stage 2*

- __Preprocessing__. In the dataset we are given patients and records of their signals: accelerometer, BVP, EDA. We combine these signals into one, then segment them so that the segment indexes coincide with the beginning and end in time.

- __Normalization__. We augment the segments to reduce class imbalance. To do this, we add noise to segments with epilepsy. Next, we balance the classes by getting rid of segments with no epilepsy so that the number of segments in one class matches the number of segments in the other class.

- __Submission to the model__. In this step, we add the spectra of these signals and time to the signal matrix. 
Finally, we feed these matrices into the model for training.

*Stage 3*
