
# **License Plate Detection using YOLOv8**

This project demonstrates license plate detection using YOLOv8, starting from downloading the Robo dataset, training the YOLOv8 model, running inference, and applying image processing techniques with OpenCV.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Dataset](#dataset)
5. [YOLOv8 Training](#yolov8-training)
6. [Inference](#inference)
7. [OpenCV Image Processing](#opencv-image-processing)
8. [Usage](#usage)

---

## **Overview**

License plate detection is a crucial task in automated systems such as traffic monitoring, toll systems, and parking lot management. This project utilizes YOLOv8 for real-time license plate detection, combined with OpenCV for further image processing.

---

## **Features**

- Real-time license plate detection using YOLOv8.
- Preprocessing images with OpenCV for enhanced detection.
- Flexible inference on images, videos, and live streams.
- Step-by-step instructions for training and inference.

---

## **Requirements**

Before starting, ensure you have the following dependencies installed:

- Python 3.8 or later
- Required Python libraries:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `matplotlib`

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## **Dataset**

We use the Robo dataset for training, which contains annotated images of license plates.

### **Steps to Download the Dataset:**
   ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="lu08WwdIJ89fRfTh16mT")
    project = rf.workspace("kanwal-masroor-gv4jr").project("yolov7-license-plate-detection")
    version = project.version(3)
    dataset = version.download("yolov8")
   ```
   Modify the `data.yaml` file to match your dataset paths:
   ```yaml
   train: ../train/images
   val: ../valid/images
   test: ../test/images
   nc: 1
   names: ['licenseplate']
   ```
---

## **YOLOv8 Training**

1. **Install YOLOv8:**
   ```bash
   pip install ultralytics
   ```

2. **Train the YOLOv8 Model:**
   ```python
   from ultralytics import YOLO

   model = YOLO('yolov8n.yaml')  # Load YOLOv8 configuration
   model.train(data='/path/to/data.yaml', epochs=15, imgsz=640)
   ```

3. **Save and Export the Model:**
   After training, the best model will be saved in the `runs/detect/train/weights/` directory.

---

## **Inference**

Use the trained model to detect license plates on images, videos, or streams.

## **OpenCV Image Processing**
   ```python
      python inference.py
      # See the out put in output folder under assignment.
   ```
![License plate masking Images](images/example_image.jpg)

## **Usage**

### **Training the Model**
1. Place your dataset in the specified structure.
2. Update the `data.yaml` file with dataset paths.
3. Run the training script.

### **Inference**
1. Use the trained weights for inference.
2. Visualize results or save the output.
