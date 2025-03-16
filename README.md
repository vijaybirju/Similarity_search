# **Semantic Similarity Search**  

A Natural Language Processing (NLP) project to quantify the degree of similarity between two text paragraphs using **TF-IDF + Cosine Similarity** and **Sentence-Transformer embeddings**. The project provides a scalable API that computes semantic similarity scores efficiently.  

## **Table of Contents**  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Technologies Used](#technologies-used)  
- [Contributing](#contributing)  
- [License](#license)  

## **Overview**  

This project implements **semantic similarity scoring** between text pairs using two methods:  

1. **TF-IDF + Cosine Similarity:** A statistical approach that converts text into numerical vectors based on word importance and measures similarity using cosine distance.  
2. **Sentence-Transformers:** A deep learning-based approach that generates dense vector representations of sentences, capturing contextual meaning.  

The project includes a **FastAPI-based server** that exposes an API endpoint to compute similarity scores dynamically. The model is deployed on **Render** using Hugging Face inference for transformer-based scoring.  

---

## **Features**  

✅ Computes similarity between text pairs using **two different approaches**  
✅ REST API for real-time similarity computation  
✅ Lightweight **TF-IDF** method for quick scoring  
✅ **Sentence-Transformers** for deep semantic understanding  
✅ Scalable deployment using **Render & FastAPI**  

---

## **Installation**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/vijaybirju/similarity-search.git
cd similarity-search
```

## 2. Create a Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. Run the API Server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

2. API Request Example \
POST /prediction \
Computes the similarity between two input texts.
Request:
```bash
{
  "text1": "nuclear body seeks new tech .......",
  "text2": "terror suspects face arrest ........"
}
```
Response:
```bash
{
    "text1": "broadband access expands",
    "text2": "broadband is growing",
    "similarity score": 0.7656
}
```


## Semantic Text Similarity API

This project implements a **text similarity prediction API** using **TF-IDF** and **Hugging Face Sentence Transformers**. The API is built with **FastAPI** and deployed on a cloud service. It computes the similarity between two text paragraphs and returns a score between **0 (dissimilar)** and **1 (highly similar).**

## Features
- Preprocessing text data using **TF-IDF**.
- Utilizing **Hugging Face Sentence Transformers** for embeddings.
- Computing similarity using **cosine similarity**.
- FastAPI-based REST API for real-time inference.
- Cloud deployment for scalable access.

---


## Project Structure
```bash
similarity-search/
│── data/                # Raw, processed, and external data
│── docs/                # Documentation files
│── models/              # Trained models
│── notebooks/           # Jupyter notebooks for EDA and experimentation
│── reports/             # Analysis reports and visualizations
│── src/                 # Source code for the project
│   ├── data/            # Data processing scripts
│   ├── features/        # Feature engineering scripts
│   ├── models/          # Training and inference scripts
│   ├── visualization/   # Data visualization scripts
│   ├── api.py           # FastAPI-based API server
│── requirements.txt     # Python dependencies
│── setup.py             # Setup script for packaging
│── README.md            # Project documentation
```

## Technologies Used
* Python (FastAPI, Scikit-learn, Transformers, Uvicorn)
* NLP Models (TF-IDF, Sentence-Transformers)
* Deployment (Render, Hugging Face Inference API)




## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.


