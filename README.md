# Miner Safety Helmet – Machine Learning Module

## Project Overview

This repository contains the machine learning module developed for an intelligent miner safety helmet. The system uses a Multi-Layer Perceptron (MLP) neural network to classify a miner's condition based on sensor measurements, including acceleration, gyroscope readings, heart rate, SpO2, temperature, and humidity.

The trained model is converted into an INT8 TensorFlow Lite model to reduce memory requirements and support lightweight inference on an ESP32-based embedded platform.

**Model Architecture:** 10 → 32 → 16 → 4  
**Input Features:** 10 sensor parameters  
**Output Classes:** 4  
**Deployment Format:** TensorFlow Lite INT8  
**Model Size:** Approximately 4.8 KB

## 1. System Workflow

1. Sensor dataset generation and collection.
2. Data preprocessing and validation.
3. Feature scaling and missing-value handling.
4. MLP model training.
5. Performance evaluation using unseen test samples.
6. TensorFlow Lite conversion and INT8 quantization.
7. Model integration for ESP32-based inference.

## 2. Dataset Description

The initial development dataset consists of 8,000 synthetically generated sensor records. It was created using `generate_dataset.py` to establish and validate the complete machine learning pipeline before collecting actual sensor measurements from the helmet prototype.

The dataset includes four operating conditions:

| Class ID | Classification | Samples |
|---|---|---:|
| 0 | NORMAL | 4,439 |
| 1 | FALL | 1,187 |
| 2 | HEALTH_ALERT | 1,190 |
| 3 | ENVIRONMENT_ALERT | 1,184 |

### Sensor Input Parameters

The model uses the following ten features in a fixed order:

| Parameter | Unit |
|---|---|
| AccX, AccY, AccZ | g |
| GyroX, GyroY, GyroZ | deg/s |
| Heart Rate | bpm |
| SpO2 | % |
| Temperature | °C |
| Humidity | %RH |

The development dataset incorporates missing values, sensor-range errors, overlapping operating conditions, and a small proportion of label variations to evaluate the preprocessing and classification pipeline.

**Dataset limitation:** These records represent simulated sensor conditions and are not measurements collected from actual miners. The results obtained from this dataset should therefore be interpreted as development-stage model performance rather than validated real-world safety performance.

## 3. Machine Learning Architecture

A Multi-Layer Perceptron (MLP) is implemented for multi-class classification.

| Layer | Neurons |
|---|---:|
| Input layer | 10 |
| Hidden layer 1 | 32 |
| Hidden layer 2 | 16 |
| Output layer | 4 |

The network learns relationships between sensor measurements and the four predefined operating conditions.

The model is trained using TensorFlow/Keras and evaluated using a separate test dataset.

## 4. Model Performance

The dataset is divided using stratified sampling, with 80% allocated for training and 20% for testing.

### Test Dataset Results

| Model | Accuracy | FALL Recall | Model Size |
|---|---:|---:|---:|
| Keras Float32 | 89.75% | 88.6% | — |
| TFLite Float32 | 89.75% | 88.6% | 5,980 bytes |
| TFLite INT8 | 89.81% | 89.0% | 4,856 bytes |

### Class-wise F1 Scores

| Classification | F1 Score |
|---|---:|
| NORMAL | 0.915 |
| FALL | 0.894 |
| HEALTH_ALERT | 0.836 |
| ENVIRONMENT_ALERT | 0.890 |

INT8 quantization reduces the model size while maintaining classification performance close to the original floating-point model.

These results are based exclusively on the synthetic test dataset.

## 5. Fall Detection and Alert Threshold

For safety-oriented inference, the model uses the predicted FALL probability to determine whether an alert should be triggered.

The implemented alert condition is:

`P(FALL) >= 0.30`

This threshold is intended to increase fall-event sensitivity compared with selecting only the highest-probability class.

| Detection Method | FALL Recall | False Alarms |
|---|---:|---:|
| Argmax classification | 88.6% | 23 |
| 0.30 probability threshold | 92.4% | 39 |
| 0.20 probability threshold | 93.2% | 55 |
| 0.10 probability threshold | 97.0% | 84 |

The threshold demonstrates the trade-off between missed fall events and false alerts. The appropriate threshold must be established through testing with actual sensor data before deployment.

## 6. Model Analysis

Permutation-based feature importance indicates that heart rate, temperature, humidity, acceleration, and SpO2 contribute to the overall classification performance.

For fall classification, AccX and AccZ show comparatively stronger contributions.

The model's performance is influenced by the distributions and assumptions used to generate the development dataset. Actual sensor behaviour, individual differences, environmental conditions, and motion artefacts may produce different results.

## 7. ESP32 Deployment

The trained neural network is converted into an INT8 TensorFlow Lite model for embedded inference.

The ESP32 integration files include:

- `helmet_inference.ino` – Embedded inference implementation.
- `helmet_model_data.h` – Model parameters and quantized model data.
- `helmet_scaler.h` – Feature normalization parameters.

The inference implementation includes the input quantization and output dequantization operations required by the INT8 model.

The current Arduino sketch contains placeholder sensor-reading functions. Actual sensor drivers and hardware integration must be completed before deployment.

The model conversion and inference pipeline have been verified in Python using the TensorFlow Lite interpreter. Hardware-level execution has not yet been validated.

## 8. Project Structure

```text
helmet_ml/
│
├── data/
│   └── helmet_sensor_dataset.csv
│
├── models/
│   ├── helmet_mlp.keras
│   ├── helmet_model.tflite
│   ├── helmet_model_float.tflite
│   ├── scaler.pkl
│   ├── imputer.pkl
│   └── scaler_params.json
│
├── reports/
│   ├── training_curves
│   ├── confusion_matrix
│   ├── feature_importance
│   └── evaluation_report.txt
│
├── esp32/
│   ├── helmet_model_data.h
│   ├── helmet_scaler.h
│   └── helmet_inference.ino
│
├── generate_dataset.py
├── preprocess.py
├── train_mlp.py
├── evaluate_model.py
├── convert_tflite.py
├── requirements.txt
└── run_all.sh
```

## 9. Execution Instructions

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Execute the complete pipeline:

```bash
./run_all.sh
```

Alternatively, execute each stage individually:

```bash
python generate_dataset.py
python preprocess.py
python train_mlp.py
python evaluate_model.py
python convert_tflite.py
```

## 10. Future Development

The following improvements are planned for further development:

- Collect actual sensor measurements from the helmet prototype.
- Validate the model under different working and environmental conditions.
- Incorporate time-window-based motion analysis for fall detection.
- Evaluate sensor noise, drift, and measurement reliability.
- Perform ESP32 hardware-level inference testing.
- Optimize alert thresholds using real-world validation data.
- Implement a rule-based backup mechanism for critical safety events.

## 11. Development Status

The current implementation establishes a complete machine learning development pipeline, from dataset preparation and model training to INT8 conversion and embedded inference preparation.

The system is currently at the prototype development stage. Real-world miner data collection, physical hardware validation, and safety-performance evaluation remain necessary before operational deployment.

**Note:** This project is an experimental safety-monitoring prototype and is not a certified mine-safety device.
