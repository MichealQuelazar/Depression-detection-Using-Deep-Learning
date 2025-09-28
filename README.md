# Depression-detection-Using-Deep-Learning  
**Mini Project Work for BTAM1311 (Deep Learning)**  

<img width="1817" height="924" alt="Screenshot 2025-09-17 213625" src="https://github.com/user-attachments/assets/70137bc5-05f7-4ec2-ae56-bcadaa07bec4" />

This project applies **Convolutional Neural Networks (CNNs)** to detect signs of depression from facial expression images.  
It includes dataset preparation, model training, evaluation, a project report, and a Flask web application for deployment. 

## 🚀 Live Demo
Check out the project here: [Live Demo](https://depression-detection-using-deep-learning.onrender.com/)


---

## Features

- Original development notebook: `notebooks/Depression-detection-Using-Deep-Learning.ipynb`  
- Training script alternative: `train.py`  
- Flask web application for image-based predictions  
- Dataset included in `notebooks/dataset/`  
- Project report included in `notebooks/report/`  

---

## Project Structure
```bash
Depression-detection-Using-Deep-Learning/  
│── app.py                  # Flask web app  
│── train.py                # Training script  
│── depression_model.h5     # Saved model  
│── requirements.txt        # Dependencies  
│── README.md               # Documentation  
│  
├── templates/              # HTML templates  
│   └── index.html  
│  
├── static/                 # Static assets (CSS, images)  
│   └── style.css  
│  
└── notebooks/  
    ├── Depression-detection-Using-Deep-Learning.ipynb  # Original project notebook  
    ├── dataset/  # Dataset for training/testing
    └── report/             # Project report  

---
```

## Installation & Setup  

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Depression-detection-Using-Deep-Learning.git
```
### Navigate to cloned repoistory
```bash
cd Depression-detection-Using-Deep-Learning
```
 ### Create a virtual environment
 ```bash
python -m venv venv
```
 ### Activate your virtual environment
 ```
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Run the Flask app
```bash
python app.py
```

### Download the original dataset used from Kaggle: 
```bash
https://www.kaggle.com/datasets/khairunneesa/depression-dataset-on-facial-ecpression-images
```

 
