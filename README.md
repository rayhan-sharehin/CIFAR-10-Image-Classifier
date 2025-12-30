# CIFAR-10 Image Classification System

**Author:** Nazmul Islam Rayhan  
**Role:** AI-ML Engineer


## Project Overview
This project implements a **CIFAR-10 Image Classification System** using **PyTorch**.  
It classifies images into 10 classes:  

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

The system includes:

- **Data loading & preprocessing** using `torchvision.datasets.CIFAR10`.
- **Deep learning model** built with PyTorch (CNN architecture).
- **Training pipeline** with GPU/CPU support.
- **Evaluation** with accuracy metrics and sample predictions.
- **Deployment** using **Docker** for interactive web demo.

## Features

- Predict classes for user-uploaded images.
- Interactive Gradio interface with live demo.
- Supports both CPU and GPU inference.
- Offline-safe model loading for environments without internet.


