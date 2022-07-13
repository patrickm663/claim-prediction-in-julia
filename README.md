# claim-prediction-in-julia
A simple example applying Julia's Flux libary to an auto-insurance dataset.

## Purpose
This example (and more to come) aims to see how an artifical neural network fares in a claim prediction task. Most Flux examples seem to either show very simple examples (like predicting the function f(x) = 4x + 2) or focus on image recognition tasks (like the well-known MINST) and then start focussing on more complex neural network structures. Through this example, we hope to show a middle-ground and introduce users to other interesting datasets after exhausting 'Boston' et. al.

## Overview of the Data
The choice of data is a collection of 67'856 motor vehicle insurance policies recorded by an Australian short-term insurer in 2004. The dataset comprises of 4'624 policyholders that had at least one claim, making it fairly unbalanced with a target class making up just under 7% of the dataset.

The dataset is found in R's 'CASDatasets' package, titled 'ausprivauto0405' (source: P. De Jong and G.Z. Heller (2008), Generalized linear models for insurance data, Cambridge University Press. _Retrieved from 'CASDatasets' version 1.0-11_). 

The dataset consist of 9 columns comprising of:
- **Exposure:** Number of policy years
- **VehValue:** Vehicle value in 000s of AUD
- **VehAge:** Age group of the vehicle
- **VehBody:** Vehicle body group (e.g. hatchback, sedan)
- **Gender:** Gender of the policyholder
- **DrivAge:** Age of the policyholder
- **ClaimOcc:** A binary indicator of whether or not a claim has occured (_target variable_)
- **ClaimNb:** Number of claims that occured per policyholder in the year (max = 4) (_excluded as features_)
- **ClaimAmount:** Sum of claim payments in AUD (_excluded as features_)

When dummy encoded (dropping one feature per class as a baseline), we get 23 features feeding into our model. The train/test split is 70-30 using partitioned random sampling to maintain a similar distribution of the target variable.

## Model Architecture
The ANN comprises of 23 input neurons, one hidden layer with 10 neurons and a tanh activation function, and a single output neuron with a sigmoid activation function. ADAM is used as its optimiser using default parameters. The ANN is trained on 2'000 epochs.

## TODO
Further development is underway to handle class imbalance. Current model results suggest the model only picks the majority class.
