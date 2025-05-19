
# Face Recognition System using OpenCV & Python

This is a simple real-time face recognition project using Python and OpenCV.  
It allows you to:

- Capture face images from webcam  
- Train a face recognizer using LBPH algorithm  
- Recognize faces in real-time  
- Display names and bounding boxes over detected faces  

---

## Technologies Used

- Python 3  
- OpenCV  
- NumPy  

---

## Features

✅ Face detection using Haar Cascade  
✅ Face recognition with LBPH  
✅ Save captured face images in dataset folder  
✅ Works with live webcam  
✅ Simple command-line interface  

---

## Folder Structure

```

📁 dataset/
└── person\_name/
├── image\_0.jpg
├── image\_1.jpg
└── ...
📄 face_recognition.py

````

---

## How to Run

1. Install the requirements:
```bash
pip install opencv-python opencv-contrib-python numpy
````

2. Run the script:

```bash
python face_recognition.py
```

3. Select an option:

```
1. Train
2. Recognize
q. Quit
```

---

## Training a Face

* Choose option 1: "Train"
* Enter a name (e.g., "Yasir")
* It will capture 100 grayscale face images and store them in dataset/Yasir/
* Then, it trains the model on all available folders

---

## Recognizing Faces

* Choose option 2: "Recognize"
* The system will open the webcam and recognize faces based on previously trained data

---

## Future Improvements

* Save and load trained model to avoid re-training
* Add GUI interface using Tkinter or PyQt
* Improve face detection using deep learning
* Support less processing power consumption feature for multi faces

---
