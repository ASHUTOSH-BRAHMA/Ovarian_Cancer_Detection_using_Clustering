<img width="1024" height="1024" alt="Gemini_Generated_Image_5w8fel5w8fel5w8f" src="https://github.com/user-attachments/assets/764c1e06-6172-4415-a034-88f721137147" /># Ovarian Cancer Detection Using DenseNet121 & Unsupervised Clustering

This project presents a fully automated deep-learning pipeline for classifying ovarian cancer images using DenseNet121, K-Means clustering, and transfer learning. Designed for unlabeled medical datasets, the system generates pseudo-labels through clustering and trains a powerful classifier capable of detecting cancer patterns without manually annotated data.

🚀 Features

🔍 Unsupervised clustering (K-Means) to auto-label unlabeled medical images
🧠 Feature extraction using pretrained DenseNet121 (ImageNet)
🗂️ Automatic dataset re-organization into cluster-based class folders
📦 Train/Validation split using Keras ImageDataGenerator
🏋️ Two-stage training:
Frozen DenseNet121 backbone
Full fine-tuning with low learning rate
🛑 Smart callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
💾 Saves best and final models (best_model.h5, final_ovarian_densenet121.h5)
⚡ GPU-optimized pipeline for Kaggle

📁 Project Structure
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
└── README.md!
[Uploading Gemini_Generated_Image_5w8fel5w8fel5w8f.png…]()


🧠 Pipeline Overview
1. Feature Extraction
DenseNet121 extracts 1024-dimensional embeddings from each image.
2. K-Means Clustering
Images are grouped into clusters forming pseudo-labels.
3. Dataset Construction
Images are automatically moved into cluster_0, cluster_1, etc.
4. Model Training
A DenseNet121 classifier with a custom head is trained with:
Adam optimizer
Categorical cross-entropy
80/20 train-validation split
5. Fine-Tuning
DenseNet121 is unfrozen and fine-tuned with a lower learning rate.

▶️ Usage
Run the notebook:
ovarian_cancer_detection.ipynb

Make sure your dataset directory follows this format:
/kaggle/input/your-dataset/
The script will:
Auto-extract features
Cluster images
Build dataset
Train and fine-tune DenseNet121
Save final models

🧪 Output Files
best_model.h5 – Best validation accuracy
final_ovarian_densenet121.h5 – Fully fine-tuned model
clustered_data/ – Auto-generated dataset
Optional: prediction CSV files and visual outputs

🛠️ Tools & Libraries
TensorFlow / Keras
DenseNet121
Scikit-learn
NumPy
TQDM
