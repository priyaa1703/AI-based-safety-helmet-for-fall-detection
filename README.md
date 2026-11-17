# AI-Based Miner Safety Helmet for Fall Detection

## Project Overview

The AI-Based Miner Safety Helmet is an intelligent safety monitoring system designed to monitor miners' physical conditions and surrounding environmental parameters using an ESP32-based sensor interface and a machine learning classification model.

The system integrates motion sensors, physiological sensors and environmental sensors to identify abnormal operating conditions, detect potential fall events and generate safety alerts.

A lightweight Multi-Layer Perceptron (MLP) model is implemented to classify sensor inputs into four categories. The trained model is optimized using INT8 quantization and converted into TensorFlow Lite format for resource-efficient embedded inference.

### Key Features

- Real-time motion monitoring using accelerometer and gyroscope measurements.
- Heart rate and SpO2 monitoring.
- Temperature and humidity monitoring.
- Machine learning-based fall detection.
- Health and environmental condition classification.
- Lightweight INT8 model for embedded deployment.
- ESP32-based sensor interfacing and inference architecture.

## 1. Hardware Architecture

The hardware system uses an ESP32 microcontroller as the primary processing unit. Multiple sensors are interfaced to acquire motion, physiological and environmental measurements.

| Component | Function |
|---|---|
| ESP32 | Main microcontroller and processing unit |
| MPU6050 | 3-axis acceleration and gyroscope measurements |
| MAX30102 | Heart rate and SpO2 monitoring |
| Temperature Sensor | Monitors surrounding temperature |
| Humidity Sensor | Measures environmental humidity |

### Sensor Parameters

The system processes ten sensor parameters:

| Sensor Category | Parameters | Unit |
|---|---|---|
| Acceleration | AccX, AccY, AccZ | g |
| Angular Velocity | GyroX, GyroY, GyroZ | deg/s |
| Physiological | Heart Rate | bpm |
| Physiological | SpO2 | % |
| Environmental | Temperature | °C |
| Environmental | Humidity | %RH |

These measurements form the input features for the machine learning classification model.

## 2. Machine Learning Architecture

A Multi-Layer Perceptron (MLP) neural network is implemented to classify the monitored conditions based on the sensor parameters.

**Network Architecture:**

`10 → 32 → 16 → 4`

| Layer | Neurons | Description |
|---|---:|---|
| Input Layer | 10 | Sensor measurements |
| Hidden Layer 1 | 32 | Feature learning |
| Hidden Layer 2 | 16 | Feature representation |
| Output Layer | 4 | Condition classification |

### Classification Categories

| Class ID | Output Class | Description |
|---|---|---|
| 0 | NORMAL | Normal operating condition |
| 1 | FALL | Potential fall event |
| 2 | HEALTH_ALERT | Abnormal physiological readings |
| 3 | ENVIRONMENT_ALERT | Abnormal environmental conditions |

The model is implemented using TensorFlow and Keras. Data preprocessing includes missing-value handling, feature scaling and stratified training and testing.

## 3. Model Training and Evaluation

The dataset contains 8,000 sensor-parameter records and is divided into training and testing subsets using an 80:20 stratified split.

### Classification Performance

| Classification | F1 Score |
|---|---:|
| NORMAL | 91.5% |
| FALL | 89.4% |
| HEALTH_ALERT | 83.6% |
| ENVIRONMENT_ALERT | 89.0% |

### Model Comparison

| Model | Accuracy | FALL Recall | Model Size |
|---|---:|---:|---:|
| Keras Float32 | 89.75% | 88.6% | — |
| TFLite Float32 | 89.75% | 88.6% | 5,980 bytes |
| TFLite INT8 | 89.81% | 89.0% | 4,856 bytes |

The INT8 quantized model achieves comparable classification performance while reducing the model size to approximately 4.8 KB.

*Performance figures represent evaluation on the available model test dataset.*

## 4. Fall Detection and Alert Mechanism

The trained model estimates the probability of each classification category.

For fall detection, an alert condition is implemented using a probability threshold:

`P(FALL) >= 0.30`

This approach prioritizes fall-event sensitivity rather than relying exclusively on the highest-probability output class.

| FALL Probability Threshold | Recall | False Alarms |
|---|---:|---:|
| Argmax classification | 88.6% | 23 |
| 0.30 | 92.4% | 39 |
| 0.20 | 93.2% | 55 |
| 0.10 | 97.0% | 84 |

The threshold provides a trade-off between missed fall events and false alerts.

## 5. Embedded Deployment – ESP32

To enable lightweight inference, the trained MLP model is converted into TensorFlow Lite format and quantized to INT8.

The embedded implementation includes:

- Quantized neural network parameters.
- Input feature normalization.
- INT8 input quantization.
- Model inference.
- Output dequantization and class probability evaluation.
- Fall detection threshold evaluation.

### ESP32 Integration Files

| File | Description |
|---|---|
| `helmet_inference.ino` | ESP32 inference implementation |
| `helmet_model_data.h` | Quantized neural network model data |
| `helmet_scaler.h` | Feature scaling parameters |

The inference implementation is structured to receive sensor readings through the ESP32 sensor interface and process them using the optimized model.

## 6. Software and Technologies

| Category | Technologies |
|---|---|
| Microcontroller | ESP32 |
| Programming | Python, C/C++ |
| Machine Learning | TensorFlow, Keras |
| Model Optimization | TensorFlow Lite, INT8 Quantization |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Embedded Development | Arduino IDE |
| Version Control | Git, GitHub |

## 7. Project Workflow

```text
Sensor Acquisition
       |
       v
ESP32 Sensor Interface
       |
       v
Sensor Data Processing
       |
       v
Feature Preprocessing
       |
       v
MLP Classification Model
       |
       v
Condition Classification
       |
       +-------------------+
       |                   |
       v                   v
  FALL Detection      Health / Environmental
       |                   |
       v                   v
  Safety Alert          Alert Output
```

## 8. Repository Structure

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

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Execute the Complete ML Pipeline

```bash
./run_all.sh
```

### Execute Individual Modules

```bash
python generate_dataset.py
python preprocess.py
python train_mlp.py
python evaluate_model.py
python convert_tflite.py
```

## 10. Future Enhancements

- Integration of temporal sensor analysis for improved fall detection.
- Optimization of sensor sampling and filtering.
- Improved classification using real-world sensor measurements.
- Further reduction of embedded memory and computational requirements.
- Hardware-level inference and performance validation.
- Development of additional safety alert mechanisms.

## Conclusion

The AI-Based Miner Safety Helmet combines embedded sensor interfacing with machine learning-based condition classification to support continuous monitoring of miner safety parameters.

By integrating motion, physiological and environmental measurements with an optimized MLP model, the system provides a compact approach to fall detection and abnormal-condition monitoring. INT8 quantization further enables deployment of the classification model on resource-constrained microcontroller platforms such as the ESP32.

**Note:** The reported classification metrics are based on the model evaluation dataset. They should not be interpreted as validated real-world fall-detection performance.
