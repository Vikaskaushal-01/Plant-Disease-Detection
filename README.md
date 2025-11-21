# 🌿 Plant Disease Detection Using Deep Learning

This project is a machine learning powered plant disease detection system that identifies plant leaf diseases using a trained TensorFlow/Keras model. It also provides precautionary steps based on the detected disease.

---

## 🚀 Features

- Detects multiple crop diseases from leaf images
- Shows prediction confidence scores
- Provides suggested precautions for each disease
- Flask-based backend API for predictions
- Supports JPG, PNG, JPEG images

---

## 🧠 Model Details

- Framework: TensorFlow / Keras
- Input Image Size: 224x224
- Output: Disease class with probability

---

## 📁 Project Structure

plant-disease-detector/
│── model/your_trained_model.h5
│── uploads/
│── app.py
│── requirements.txt
│── README.md

---

## 📦 Installation

Run the following commands in terminal or CMD:

pip install -r requirements.txt

If you don’t have requirements.txt, manually install:

pip install tensorflow flask numpy pillow

---

## ▶️ Run the Application

Start server using:

python app.py

You should see:

Running on http://127.0.0.1:5000/

---

## 🖼️ Predicting an Image

1. Place your test image inside the `uploads/` folder
2. Call prediction API:

Example URL:
http://127.0.0.1:5000/predict?image=test.jpg

---

## 📌 Sample Output

{
  "image": "uploads\\test.JPG",
  "precautions": [
    "Isolate affected plants to prevent spread.",
    "Remove and destroy infected leaves/plant parts.",
    "Improve airflow and reduce overhead watering.",
    "Rotate crops and avoid planting same crop repeatedly in same soil.",
    "Consult agricultural expert for treatment."
  ],
  "predictions": [
    { "label": "Tomato_Bacterial_spot", "probability": 0.28 },
    { "label": "Pepper__bell___Bacterial_spot", "probability": 0.27 },
    { "label": "cedar_apple_rust", "probability": 0.23 }
  ]
}

---

## 🛠️ Future Improvements

- Add frontend UI
- Mobile app support
- Real-time camera detection
- Expand dataset for better accuracy

---

## ❤️ Contributions

Contributions, issues, and feature requests are welcome.
Feel free to fork and submit pull requests.

---

## 📄 License

This project is open-source and available under the MIT License.

