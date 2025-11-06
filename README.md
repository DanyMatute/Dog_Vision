# 🐶 End-to-End MultiClass Dog Breed Classification
This project uses a **deep learning–based image classifier** built with **TensorFlow 2.0** and **Keras**, fine-tuned to identify **120 dog breeds** from Kaggle’s Dog Breed Identification dataset.

### **📌 Why this approach?**

* The dataset contains **unstructured image data**, which is best handled using **Convolutional Neural Networks (CNNs)**.
* Instead of training a CNN from scratch (which is slow and requires a huge dataset), we use **transfer learning**—starting with a model pre-trained on ImageNet and adapting it to dog breeds.
* This significantly reduces training time and improves performance.

---

## 🧠 **Model Architecture**

The core of the model is **MobileNetV2**, a lightweight, efficient CNN architecture designed for image classification:

* **Backbone:** `tf.keras.applications.MobileNetV2`

  * Pretrained on ImageNet
  * `include_top=False` → removes final classification head
  * Extracts high-level visual features from dog images
* **Head:**

  * `GlobalAveragePooling2D()` → reduces feature map to a compact vector
  * `Dense(120, activation="softmax")` → outputs probability distribution over the 120 breeds

**Full architecture (simplified):**

```
Input (224x224x3)
↓
MobileNetV2 backbone (feature extractor)
↓
Global Average Pooling
↓
Dense layer (120 units, softmax)
↓
Breed probabilities
```

This design keeps the model small, fast, and accurate.

---

## 🎯 **Training Procedure**

### **Data Pipeline**

* Images loaded as TensorFlow `tf.data` pipelines
* Preprocessing:

  * Resize to `224 × 224`
  * Apply normalization consistent with MobileNetV2
  * Shuffle & batch data for training

### **Training Setup**

* **Loss:** Categorical Crossentropy
  (ideal for multi-class probability-based classification)
* **Optimizer:** Adam
  (adaptive learning rate, good general-purpose choice)
* **Metrics:** accuracy
* **Callbacks used:**

  * **TensorBoard** → real-time training visualizations
  * **EarlyStopping** → prevent overfitting by stopping when validation accuracy stalls

### **Epochs**

Trained for up to 100 epochs, but early stopping typically halts around the plateau.

---

## 📈 **Model Outputs**

For each test image:

* The model predicts **120 probability values**, one per breed.
* Predictions are formatted into a CSV file matching Kaggle’s submission requirements:

```
id,affenpinscher,afghan_hound,...,yorkshire_terrier
image1.jpg,0.0021,0.0451,...,0.0003
...
```

---

## ✅ **Why MobileNetV2?**

* Small and efficient (works well on limited hardware)
* Fast training
* Accuracy competitive with larger models
* Pretrained on ImageNet → learned general image features
* Works extremely well for transfer learning tasks like dog breeds

---

## 🔍 **What This Model Can Do**

✔ Classify images into **120 dog breeds**
✔ Generate probability scores for each breed
✔ Provide accurate predictions on unseen test images


