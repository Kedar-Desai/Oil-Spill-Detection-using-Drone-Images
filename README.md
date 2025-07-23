
# Deep Learning Framework for Real-time Oil Spill Detection and Classification

This repository contains the implementation of a novel deep learning framework for real-time oil spill detection and classification, leveraging the YOLOv8 architecture and advanced computer vision techniques. The research aims to enhance environmental monitoring and disaster response by providing an efficient and accurate method for identifying various types of oil spills in both image and video data.

## Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Methodology](#methodology)
* [Model Architecture](#model-architecture)
* [Results](#results)
* [Future Work](#future-work)
* [Citation](#citation)
* [License](#license)
* [Contact](#contact)

## Introduction

Oil spills pose significant environmental risks, impacting ecosystems, aquatic life, and human communities. Traditional detection methods are often time-consuming and prone to errors. This project addresses the urgent need for proactive measures by integrating deep learning and computer vision technologies to revolutionize the speed and accuracy of oil spill identification, facilitating faster response times and more effective containment efforts.

## Features

* **Real-time Detection:** Utilizes YOLOv8 for efficient and timely identification of oil spills[cite: 26, 27, 268].
* **Classification:** Distinguishes between different oil spill types and natural environmental features[cite: 28].
* **High Accuracy:** Achieves impressive overall accuracy (R-score) of 0.531 and Mean Average Precision (mAP) of 0.549[cite: 13, 215].
* **Versatile Application:** Capable of detecting oil spills in both image and video data[cite: 17, 213, 255].
* **Robustness:** Enhanced through meticulous dataset curation, model training, and data augmentation techniques[cite: 13, 92].
* **Performance Visualization:** Provides insights into model performance through box loss, class loss, and confusion matrices[cite: 15, 220, 251].

## Methodology

The proposed framework involves several meticulous steps to ensure the robustness and reliability of the oil spill detection system.

1.  **Dataset Collection and Preprocessing:** Drone images and videos (250 images, 4 videos) from the Port of Antwerp Bruges were collected[cite: 16, 67, 83]. [cite_start]Data was resized to a consistent resolution, pixel values normalized, and artifacts addressed[cite: 84].
2.  **Train-Validation-Test Split:** The dataset was split into 70% for training, 20% for validation, and 10% for testing, with random shuffling to maintain representative distribution[cite: 72, 73, 74, 75, 85, 86].
3.  **Annotation:** Oil spills were meticulously annotated with bounding boxes using tools like `labelimg` [cite: 88, 89, 294] or `RectLabel` [cite: 88, 89, 295], ensuring consistency and accuracy[cite: 88, 89].
4.  **Model Training:** The YOLOv8 architecture was enhanced with data augmentation techniques to boost robustness and generalization across diverse scenarios[cite: 92]. The Adam optimization algorithm was used with defined learning rates, batch sizes, and epochs[cite: 93, 94]. Progress was monitored using key metrics like loss, with early stopping mechanisms implemented when necessary[cite: 95].
5.  **Evaluation:** Key metrics such as precision, recall, F1 score, and mean average precision (mAP), were employed[cite: 96]. A confusion matrix provided deeper insights into the model's performance on the test set[cite: 97].

## Model Architecture

The core of this framework is the YOLOv8 architecture [cite: 26, 90, 100], specifically tailored for object detection of oil spills[cite: 100]. It represents an evolution in the YOLO series and integrates various layers for optimal functionality[cite: 100, 101].

* **Backbone (CSPDarknet53):** Responsible for feature extraction from input images [cite: 102], consisting of multiple Conv2d layers and ReLU activation functions[cite: 101, 103, 104, 114].
* **Neck (PANet - Path Aggregation Network):** Enhances object detection across different scales through the aggregation of features[cite: 115]. This is accomplished by employing Conv2d layers and ReLU activation functions[cite: 116].
* **Head (YOLO head):** Predicts bounding boxes, objectness scores, and class probabilities [cite: 117], incorporating YOLO layers with Conv2d layers, ReLU activation functions, and sometimes MaxPool2D layers[cite: 208, 209].
* **SPP (Spatial Pyramid Pooling):** Captures contextual information at multiple scales, enhancing the model's capacity to comprehend intricate patterns[cite: 210].
* **CSPNet (Cross-Stage Partial Network):** Contributes to improved feature fusion across different stages of the network, facilitating more effective information flow[cite: 211].
* **Output Layer:** Comprising Conv2d layers and ReLU activation functions, finalizes the architecture, generating predictions including bounding boxes and class probabilities[cite: 112, 212].

[cite_start]This cohesive architecture is designed for real-time object detection [cite: 213], making it particularly apt for tasks such as oil spill detection in drone images and videos[cite: 213].

## Results

The YOLOv8 model, upon training, achieved an overall accuracy (R-score) of 0.531 and a Mean Average Precision (mAP) of 0.549[cite: 215]. Performance for sheen detection exhibited a lower value of 0.4 [cite: 14, 216], which impacted the overall performance and accuracy of the model[cite: 216].

Visualizations of Box loss, Class loss, and Distribution Focal loss for both training and validation data [cite: 119, 120, 121, 166, 167, 168, 220], along with performance graphs [cite: 122, 123, 169, 170, 221][cite_start], consistently show decreasing losses and improved accuracy over epochs[cite: 15, 222].

The table below summarizes the model's performance:

| Class     | Images | Instances | Box   | R     | mAP50 | mAP50-95 |
| --------- | ------ | --------- | ----- | ----- | ----- | -------- |
| ALL       | 24     | 49        | 0.793 | 0.531 | 0.549 | 0.432    |
| OBJECT    | 24     | 1         | 1     | 0     | 0     | 0        |
| RAINBOW   | 24     | 3         | 0.723 | 0.891 | 0.913 | 0.693    |
| SHEEN     | 24     | 15        | 0.729 | 0.4   | 0.475 | 0.373    |
| TRUECOLOR | 24     | 30        | 0.729 | 0.833 | 0.81  | 0.664    |

*Note: Performance for sheen detection exhibited a lower mAP of 0.475, which impacted overall model performance and accuracy[cite: 216, 219].*

A confusion matrix visually represented through a heatmap offers a clear depiction of the object detection rate [cite: 249, 251, 252], aiding in the nuanced evaluation of the model's strengths and areas for improvement[cite: 252]. Examples of detected objects with their accuracy values are presented to showcase the model's versatility[cite: 253, 254]. The model's robustness was also tested on drone video data, with results displayed in Figure 6[cite: 255, 263].

## Future Work

Future efforts will focus on:

* **Regional Optimization:** Delving into regional optimization and tailoring the deep learning framework to accommodate the distinctive environmental characteristics of specific geographical regions[cite: 273, 274]. This adaptation process is key to enhancing the model's performance and increasing its applicability across diverse ecosystems[cite: 274].
* **Multi-sensor Fusion:** Integrating data from various sensors, including Synthetic Aperture Radar (SAR) and optical sensors, to create a more comprehensive and resilient oil spill detection system[cite: 275, 276]. The synergistic utilization of different sensing modalities holds the potential to elevate detection accuracy and reliability, particularly in the face of fluctuating environmental conditions[cite: 277].
* **Real-time Deployment Optimization:** Concentrated effort on real-time deployment optimization is paramount[cite: 278]. This entails refining the model's speed and efficiency in the detection algorithm, ensuring its responsiveness to emerging oil spill incidents in operational scenarios[cite: 279].

These refinements aim to enhance the model's adaptability, robustness, and practical effectiveness across a spectrum of real-world applications[cite: 280, 281].

## Citation

If you find this research helpful, please consider citing our paper:

**Kedar Desai, Pranshu Patel, Rinkal Jain, Chintan Bhatt, Steve Vanlanduit, Alessandro Bruno, & Pier Luigi Mazzeo. (2024). A Deep Learning Framework for Real-time Oil Spill Detection and Classification. *CEUR Workshop Proceedings*, *3923*, CEUR-WS.org/Vol-3923/Paper_1.pdf.**

## License

This project is licensed under the Creative Commons License Attribution 4.0 International (CC BY 4.0)[cite: 35].

## Contact

* Kedar Desai
* Pranshu Patel
* Rinkal Jain
* Chintan Bhatt
* Steve Vanlanduit
* Alessandro Bruno
* Pier Luigi Mazzeo


