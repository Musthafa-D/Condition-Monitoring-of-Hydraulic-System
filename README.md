# Condition-Monitoring-of-Hydraulic-System

The task aimed to evaluate condition monitoring of hydraulic system and provide training and tuning based on the given dataset for reliable predictions about expected maintenance of each valves and other parameters through supervised learning.

A DFNN network was built for the system to train itself to predict the correct outcomes of the system based on different conditions or factors.

This task involves Fault Classification and Condition Monitoring of hydraulic system 
which plays the crucial role in predictive maintenance of industrial plants and machineries. 

Motivation: The objective of this assignment is condition monitoring and fault classification 
of a hydraulic system using data-driven machine learning methods. 

Tasks: 
   Detailed exploration and analysis of the Dataset, 
   Development of a suitable model architecture, Discussing and selection of appropriate 
  loss function, 
   Implementation of a training based on a given data loader, 
   Reusing this training for a tuning loop, 
   Detailed evaluation and comparison of the obtained results 
  Machine learning method such as Deep Feedforward Neural Network (DFNN) is chosen and 
  to be implemented using PyTorch are to be applied. This type of Neural Network is 
  developed in order to perform classification tasks based on the provided datasets. 
  Comparison of the testing results for the different cases in the DFNN to be analyzed.
  
Dataset provided was experimentally obtained with a hydraulic test rig. This test rig 
consists of a primary working and a secondary cooling-filtration circuit, connected via the 
oil tank. The system cycle repeats constant load cycles (duration 60 seconds) and 
measures process values such as pressures, volume flows and temperatures while the 
condition of four hydraulic components (Cooler, Valve, Pump and Accumulator) are 
quantitatively varied. 

The input dataset contains data obtained from 17 different sensors namely “PS1”, “PS2”, 
“PS3”, “PS4”, “PS5”, “PS6”, “EPS1”, “FS1”, “FS2”, “TS1”, “TS2”, “TS3”, “TS4”, “VS1”, 
“CP”, “CE” and “SE”. Four types of faults are superimposed with their respective grades of 
severity, provided in a label dataset named “Profile.txt”. All the input and output datasets 
contain equal number of instances (2205) while the different input sensors possess different 
attributes per instance. Hence, the input datasets can be termed as Multivariate-Time Series 
Data.
