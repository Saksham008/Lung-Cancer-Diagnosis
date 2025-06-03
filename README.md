# Lung-Cancer-Diagnosis

This project implements deep learning models for the classification of histopathological lung cancer images. It features individual CNNs like **Xception** and **VGG16**, as well as a **fusion model enhanced with CBAM (Convolutional Block Attention Module)** to improve diagnostic accuracy.

---

## 📁 Dataset

The dataset used is the **Histopathological Lung Cancer Image Dataset**, available on Kaggle.

- **Source**: [Kaggle Dataset – Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- **Classes**:
  - `aca`: Adenocarcinoma
  - `scc`: Squamous cell carcinoma
  - `n`: Normal tissue

- **Splits**:
  - Training images: 4,450
  - Validation images: 3,000
  - Test images: 3,000

Images are histopathological slices stained with H&E and resized for deep learning training.

---

## 📦 Project Structure
lung-cancer-classification/
├── LungCancer.ipynb # Preliminary notebook
├── FinalModel.py # Final training + evaluation script
├── FinalModel.ipynb # Final notebook (step-by-step analysis)
├── README.md # Documentation
├── requirements.txt # All dependencies
└── saved_models/
└── final_model_vgg16.keras # Trained fusion model with CBAM

pip install -r requirements.txt

📥 Pretrained Model (Download)
Due to GitHub's file size limit, the trained model (final_model_vgg16.keras, ~168MB) is hosted externally:
Link :- https://drive.google.com/file/d/1ddmonXM5izMCN2Q5NaqFonXmz6Xj36Ce/view?usp=drive_link


✅ **To run the project using the downloaded model**, 
execute:
streamlit run FinalModel.py


👨‍💻 Authors

Saksham Vashisth
sakshamv1111@gmail.com

Divyansh Saini 
divyanshsainimzn@gmail.com

Naman Verma
vnaman896@gmail.com
