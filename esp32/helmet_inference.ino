/*
  helmet_inference.ino
  Minimal example: read sensors -> normalise -> run the INT8 MLP -> alert.

  Feature order (MUST match training):
    [AccX, AccY, AccZ, GyroX, GyroY, GyroZ, HeartRate, SpO2, Temperature, Humidity]
     idx0   1     2     3      4      5       6          7      8            9

  Class order of the output:
    0 NORMAL, 1 FALL, 2 HEALTH_ALERT, 3 ENVIRONMENT_ALERT

  Needs a TensorFlow Lite Micro library for Arduino-ESP32 (for example
  "TensorFlowLite_ESP32" or "Chirale_TensorFlowLite"). The include paths
  below follow the usual TFLM layout; tweak if your library differs.

  The three read_*() functions are placeholders - wire in your real drivers
  (MPU6050, MAX30102, DHT22 / SHT31 ...) and keep the SAME UNITS as the
  training data: g, deg/s, bpm, %, deg C, %RH.
*/

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "helmet_model_data.h"
#include "helmet_scaler.h"

#define BUZZER_PIN 25
#define LED_PIN    26

// Alert if P(FALL) is above this, even when FALL is not the top class.
// Lower = fewer missed falls but more false alarms (see reports/evaluation_report.txt).
const float FALL_ALERT_THRESHOLD = 0.30f;

constexpr int kArenaSize = 8 * 1024;           // tiny MLP, 8 KB is plenty
alignas(16) static uint8_t tensor_arena[kArenaSize];

static tflite::MicroErrorReporter error_reporter;
static tflite::AllOpsResolver resolver;        // fine for a first test; swap for
                                               // MicroMutableOpResolver<3> (FullyConnected,
                                               // Relu, Softmax) later to save flash
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// ---------------- sensor stubs (replace with real drivers) ----------------
void read_motion(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  // e.g. MPU6050: acceleration in g, gyro in deg/s
  ax = 0.02f; ay = -0.01f; az = 0.98f;
  gx = 3.0f;  gy = -2.0f;  gz = 1.0f;
}
void read_health(float &heart_rate, float &spo2) {
  // e.g. MAX30102 (after your HR / SpO2 algorithm)
  heart_rate = 82.0f; spo2 = 97.0f;
}
void read_environment(float &temp_c, float &humidity) {
  // e.g. DHT22
  temp_c = 28.0f; humidity = 60.0f;
}
// ---------------------------------------------------------------------------

// A dead sensor should not feed garbage into the model. Same idea as the
// training-time imputation: swap in the training median.
float sanitize(float v, int idx, float lo, float hi) {
  if (isnan(v) || v < lo || v > hi) return FEATURE_FALLBACK[idx];
  return v;
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  const tflite::Model* model = tflite::GetModel(helmet_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    while (true) delay(1000);
  }
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize, &error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed (increase kArenaSize)");
    while (true) delay(1000);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("Helmet model ready");
}

void loop() {
  // 1. read sensors ------------------------------------------------------
  float f[N_FEATURES];
  read_motion(f[0], f[1], f[2], f[3], f[4], f[5]);
  read_health(f[6], f[7]);
  read_environment(f[8], f[9]);

  // sanity limits mirror VALID_RANGE in preprocess.py
  for (int i = 0; i < 3; i++) f[i] = sanitize(f[i], i, -16, 16);
  for (int i = 3; i < 6; i++) f[i] = sanitize(f[i], i, -2000, 2000);
  f[6] = sanitize(f[6], 6, 25, 220);
  f[7] = sanitize(f[7], 7, 70, 100);
  f[8] = sanitize(f[8], 8, -10, 70);
  f[9] = sanitize(f[9], 9, 0, 100);

  // 2 + 3. normalise (same mean/scale as training) and fill the input tensor
  //        in the exact feature order, quantizing to int8 as we go
  for (int i = 0; i < N_FEATURES; i++) {
    float scaled = (f[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i];
    int q = (int)roundf(scaled / INPUT_SCALE) + INPUT_ZERO_POINT;
    if (q < -128) q = -128;
    if (q > 127)  q = 127;
    input->data.int8[i] = (int8_t)q;
  }

  // 4. run the model -----------------------------------------------------
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    delay(500);
    return;
  }

  // 5. dequantize the output to probabilities -----------------------------
  float prob[N_CLASSES];
  int best = 0;
  for (int c = 0; c < N_CLASSES; c++) {
    prob[c] = (output->data.int8[c] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    if (prob[c] > prob[best]) best = c;
  }

  // 6 + 7. pick the class and trigger the alert -----------------------------
  bool fall_flag = prob[1] >= FALL_ALERT_THRESHOLD;   // safety bias towards catching falls
  bool alert = (best != 0) || fall_flag;
  int shown = fall_flag ? 1 : best;

  Serial.printf("%s  [N %.2f | F %.2f | H %.2f | E %.2f]\n",
                CLASS_NAMES[shown], prob[0], prob[1], prob[2], prob[3]);

  digitalWrite(LED_PIN, alert ? HIGH : LOW);
  digitalWrite(BUZZER_PIN, alert ? HIGH : LOW);
  // TODO: send over LoRa / WiFi / BLE to the surface station here

  delay(200);   // ~5 predictions per second
}
