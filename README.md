<img width="1024" height="769" alt="densenet" src="https://github.com/user-attachments/assets/d7077d22-b3d5-489b-ba95-1ac23555206b" /># 🧬 Ovarian Cancer Detection Using DenseNet121 & Unsupervised Clustering

<img src="[Uploading densenet.png…]()
" alt="Pipeline Diagram" width="100%"/>

This project presents a fully automated deep-learning pipeline for classifying ovarian cancer images using DenseNet121, K-Means clustering, and transfer learning. Designed for unlabeled medical datasets, the system generates pseudo-labels through clustering and trains a powerful classifier capable of detecting cancer patterns without manually annotated data.

---

## 🚀 Features

* 🔍 **Unsupervised clustering** (K-Means) to auto-label unlabeled medical images
* 🧠 **Feature extraction** using pretrained DenseNet121 (ImageNet)
* 🗂️ Automatic dataset re-organization into cluster-based class folders
* 📦 Train/Validation split using Keras ImageDataGenerator
* 🏋️ **Two-stage training**:

  * Frozen DenseNet121 backbone
  * Full fine-tuning with a low learning rate
* 🛑 Smart callbacks: **EarlyStopping**, **ReduceLROnPlateau**, **ModelCheckpoint**
* 💾 Saves final models (`best_model.h5`, `final_ovarian_densenet121.h5`)
* ⚡ Fully GPU-optimized for **Kaggle**

---

## 📁 Project Structure

```
├── input_dataset/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── clustered_data/
│   ├── cluster_0/
│   ├── cluster_1/
│   └── ...
│
├── best_model.h5
├── final_ovarian_densenet121.h5
├── ovarian_cancer_detection.ipynb
└── README.md
```

---

## 🧠 Pipeline Overview

### **1. Feature Extraction**

DenseNet121 extracts **1024-dimensional** embeddings from each image.

### **2. K-Means Clustering**

Images are grouped into clusters to generate pseudo-labels.

### **3. Dataset Construction**

Images are automatically organized into:

```
cluster_0/
cluster_1/
...
```

### **4. Model Training**

A DenseNet121 classifier with a custom head is trained using:

* Adam optimizer
* Categorical cross-entropy
* 80/20 train-validation split

### **5. Fine-Tuning**

DenseNet121 is unfrozen and fine-tuned with a lower learning rate.

---

## ▶️ Usage

Run the notebook:

```
ovarian_cancer_detection.ipynb
```

Dataset should be placed under:

```
/kaggle/input/your-dataset/
```

The pipeline will automatically:

* Extract features
* Cluster images
* Build dataset
* Train & fine-tune DenseNet121
* Save the final models

---

## 🧪 Output Files

* `best_model.h5` — highest validation accuracy
* `final_ovarian_densenet121.h5` — fully fine-tuned model
* `clustered_data/` — generated class folders
* Optional: prediction CSVs, plots, visual outputs

---

## 🛠️ Tools & Libraries

* TensorFlow / Keras
* DenseNet121
* Scikit-learn
* NumPy
* TQDM
