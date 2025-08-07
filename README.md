# 🧠 Medical Image Disease Prediction Backend

This repository contains the **Flask Application** for a medical image analysis platform that:
- Authenticates users
- Accepts image uploads (individually or in batch)
- Predicts diseases using a CNN model
- Stores predictions in a PostgreSQL database
- Supports Excel-based patient metadata uploads
- Generates downloadable Word reports for results

---

## 🚀 Features

- ✅ User Signup/Login
- 🖼️ Image classification using a trained CNN model
- 🧾 Word report generation (includes image, prediction, and patient details)
- 📦 Batch prediction with Excel metadata matching
- 🧠 Prediction history per user
- 🔐 Secure API
- 🗑️ Temporary file cleanup handling
- 📡 RESTful API endpoints

---

## 🏗️ Tech Stack

- Python 3.8+
- Flask
- Flask-CORS
- Flask-JWT-Extended
- Flask-SQLAlchemy
- TensorFlow / Keras
- PostgreSQL
- python-docx
- dotenv
- Pillow (PIL)


