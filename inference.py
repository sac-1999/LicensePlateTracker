import cv2
from ultralytics import YOLO
import os
project_path = './assignment/'
assignmentoutput = f"{project_path}Output/"
os.makedirs(assignmentoutput, exist_ok=True)
image_folder = f"{project_path}/data/"
model =  YOLO('./trainedmodel/best.pt')
logo = cv2.imread(os.path.join(project_path, 'logo.png'))

def transform_plate_angles(image, coordinates):
    (x1, y1), (x2, y2) = coordinates
    region = image[y1:y2, x1:x2]
    region = cv2.bilateralFilter(region, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(region, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges, (x1, y1, x2, y2)

def replace_logo(image, logo, coordinates):
    edges, (x1, y1, x2, y2) = transform_plate_angles(image, coordinates)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the specified region!")
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    lx, ly, lw, lh = cv2.boundingRect(largest_contour)
    region_x1, region_y1 = x1 + lx, y1 + ly
    region_x2, region_y2 = region_x1 + lw, region_y1 + lh
    logo_resized = cv2.resize(logo, (lw, lh))
    image[region_y1:region_y2, region_x1:region_x2] = logo_resized
    return image

for image_name in os.listdir(image_folder):
    print(image_name)
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    results = model.predict(source=image, conf=0, verbose=False)
    bestconf = 0
    bestbox = None
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf[0] > bestconf:
                bestconf = box.conf[0]
                bestbox = box
    if bestbox:
        x1, y1, x2, y2 = map(int, bestbox.xyxy[0])
        class_id = int(bestbox.cls[0])
        image = replace_logo(image, logo, ((x1, y1), (x2, y2)))
        cv2.imwrite(os.path.join(assignmentoutput, image_name), image)

print(f"Processing complete. Check the output folder for results : {assignmentoutput}")

