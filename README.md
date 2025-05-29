# 🎵 Applied Machine Learning Project: Music Genre Classifier

This project implements a **music genre classification system** using audio features extracted from `.wav` files. It supports the following genres:

> **blues**, **classical**, **country**, **disco**, **hiphop**, **jazz**, **metal**, **pop**, **reggae**, **rock**

The system includes:
- A preprocessing pipeline
- A kNN baseline model and a CNN model
- A RESTful API to interact with the model
- A streamlit interface
- Unit tests to ensure reliability and correctness

---

## Dependencies

All required libraries are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

--- 

## Running the API

By default, the API will be hosted at:http://127.0.0.1:8000. In case of loading difficulties, try http://127.0.0.1:8000/docs.

```bash
uvicorn API:app --reload 
```

--- 

## Running the Streamlit

To check out the Streamlit page, use the next command:

```bash
streamlit run app_streamlit.py
```

--- 

## Running unittests

To run the unittest, use the following comand:

```bash
python -m unittest discover -s tests
```