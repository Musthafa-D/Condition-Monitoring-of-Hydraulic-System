# Condition Monitoring of Hydraulic System

## Overview

This project focuses on **condition monitoring and fault classification** of a hydraulic system using supervised machine learning methods.  
The task involves training a **Deep Feedforward Neural Network (DFNN)** to predict system conditions and faults based on multivariate time-series sensor data.

This type of predictive approach is useful for **predictive maintenance** in industrial machinery where early detection of faults can reduce downtime and costs.

---

## Dataset

The dataset was obtained from an experimental **hydraulic test rig** with measurements recorded from 17 different sensors, including pressures, flow rates, and temperatures.

The dataset contains:
- 2205 instances
- Multivariate time-series data
- Four types of faults with varying severity  
All input features and labels are provided in the dataset.

---

## Objective

- Explore and analyze the provided dataset
- Build and train a suitable deep learning model
- Select appropriate loss function and optimizer
- Train and evaluate predictive models
- Compare results across different model configurations

The goal is to develop a model capable of **reliable classification of hydraulic system conditions**.

---

## Approach

1. **Data Loading & Preprocessing**
   - Standardization / normalization
   - Train/test split

2. **Model Architecture**
   - Deep Feedforward Neural Network (DFNN) implemented in PyTorch
   - Multiple layers with non-linear activation functions

3. **Training Pipeline**
   - Supervised learning
   - Loss function and optimizer selection
   - Iterative training loop

4. **Evaluation**
   - Model performance comparison
   - Accuracy / confusion analysis

---
