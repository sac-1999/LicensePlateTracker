
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
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

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

1. Clone the dataset repository or download it from the official source:
   ```bash
   git clone https://github.com/roboflow/dataset-repo-name.git
   ```

2. Extract the dataset and structure it as follows:
   ```
   yolov7-license-plate-detection-3/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   ```

3. Modify the `data.yaml` file to match your dataset paths:
   ```yaml
   train: /path/to/train/images
   val: /path/to/valid/images
   nc: 1  # Number of classes
   names: ['license_plate']
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

### **Run Inference on an Image:**
```python
from ultralytics import YOLO

model = YOLO('/path/to/best.pt')  # Load the trained model
results = model('/path/to/image.jpg', conf=0.5)

# Visualize results
results.show()
```

### **Run Inference on a Video:**
```python
results = model('/path/to/video.mp4', conf=0.5)
results.show()
```

---

## **OpenCV Image Processing**

OpenCV is used for post-processing detected license plates.

### **Steps for Processing:**

1. **Preprocessing Image:**
   ```python
   import cv2

   img = cv2.imread('/path/to/image.jpg')
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   cv2.imshow('Processed Image', blurred)
   cv2.waitKey(0)
   ```

2. **Cropping Detected License Plates:**
   Use bounding boxes from YOLOv8 to crop and isolate the license plate.

   ```python
   for result in results.xyxy[0]:  # Bounding boxes
       x1, y1, x2, y2, conf, cls = map(int, result)
       cropped = img[y1:y2, x1:x2]
       cv2.imshow('Cropped Plate', cropped)
       cv2.waitKey(0)
   ```

---

## **Usage**

### **Training the Model**
1. Place your dataset in the specified structure.
2. Update the `data.yaml` file with dataset paths.
3. Run the training script.

### **Inference**
1. Use the trained weights for inference.
2. Visualize results or save the output.

---

## **Results**

Example outputs:

1. **Image Detection:**
   ![Image Detection Example](example_image_detection.png)

2. **Video Detection:**
   ![Video Detection Example](example_video_detection.gif)

3. **OpenCV Processed Image:**
   ![Processed Image Example](example_processed_image.png)

---

## **Contributing**

Contributions are welcome! Please submit a pull request or raise an issue for bugs or feature requests.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
