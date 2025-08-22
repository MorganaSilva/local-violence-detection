
# Table of Contents

- [Introduction](#introduction)
- [How to Run](#howtorun)
- [Results](#results)
- [Further Work](#work)

---

<a name="introduction"/>

# Introduction

This repository contains a **real-time violence detection system** that works with any standard webcam.  

The solution is built on top of **CLIP (Contrastive Language–Image Pretraining)** from OpenAI, which encodes both images and text into the same vector space. This makes it possible to analyze video frames and compare them against predefined labels in the `settings.yaml` file to classify each frame in real time.  

The pipeline runs **fully offline**, without relying on any cloud services, ensuring low latency and efficient resource usage — ideal for controlled environments or proof-of-concept systems that need real-time video inference.  

You can customize the **labels for detection** by editing the `labels` field in the `settings.yaml` file. Adding new labels with descriptive text allows the model to generalize to additional scenarios effectively.

---

<a name="howtorun"/>

# How to Run

### 1. Requirements
- **Operating System:** Windows, Linux, or macOS  
- **Python:** **Python 3.9** is recommended for best compatibility  
- **Dependencies:** Listed in `requirements.txt`

---

### 2. Install dependencies
In your terminal, run:
```bash
pip install -r requirements.txt
```

---

### 3. Run real-time detection
To start the detection pipeline using your webcam:
```bash
python webcam.py
```

Optional arguments:
- `--camera`: Camera index (default is `0` if you have only one camera).  
- `--binary-settings`: Use this flag if `settings.yaml` is configured for two labels only (`violence` / `non-violence`).  
- `--record`: Path to save the processed video. Example:
```bash
python webcam.py --record ./output.mp4
```

---

### 4. Use the model in your own code
You can also import the model into your Python scripts for static image processing:

```python
from model import Model
import cv2

model = Model()
image = cv2.imread('./your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = model.predict(image=image)
print('Prediction:', result['label'], '| Confidence:', result['confidence'])
```

---

<a name="results"/>

# Results

Here is an example of the system running in **real time** using a **Logitech C922 webcam**, operating at **1280x720 and 15 FPS**.

![Example GIF](./results/teste2.gif)

The system includes **temporal smoothing**, which checks multiple frames (default: 12) before triggering an alert, reducing false positives in live scenarios.

---

<a name="work"/>

# Further Work

Planned improvements for future iterations:
- Export the model to **ONNX** for reduced latency and better inference performance;  
- Integration with automated alert systems;  
- Dynamic threshold tuning for different environments;  
- Support for multi-camera input;  
- Web-based monitoring interface for real-time alerts.

---