### MAIN CLASS FOR CATTLE COUNTING


import os
import cv2
import xml.etree.ElementTree as ET
import projection
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import config
from computeEdges import computeCattleEdges
from maximumFlow import solveDuplicateCattle



# Get the class index and softmax_score for color classification
def getColor(cattleImg):

    if not config.color:
        return 1, 1

    # Load the pre-trained MobileNetV3 Large model
    model_path = config.color_model
    model = tf.keras.models.load_model(model_path)

    # Resize the image to the input size expected by MobileNetV3 Large (224x224) and preprocess
    img = cv2.resize(cattleImg, (224, 224))
    img = preprocess_input(img)

    # Perform inference
    prediction = model.predict(np.array([img]))
    class_index = np.argmax(prediction)
    softmax_score = prediction[0][class_index]

    return class_index, softmax_score


# Get the class index and softmax_score for state classification (0: standing up - 1: lying down)
def getState(cattleImg):

    if not config.state:
        return 1, 1

    # Load the pre-trained MobileNetV3 Large model for state classification
    model_path = config.state_model
    model = tf.keras.models.load_model(model_path)

    # Resize the image to the input size expected by MobileNetV3 Large (224x224) and preprocess
    img = cv2.resize(cattleImg, (224, 224))
    img = preprocess_input(img)

    # Perform inference
    prediction = model.predict(np.array([img]))
    class_index = np.argmax(prediction)
    softmax_score = prediction[0][class_index]

    return class_index, softmax_score


# Get the global direction degree the cattle head is pointed to
def getGlobalDirection(cattleImg, yaw):

    if not config.direction:
        return 0

    # Load the pre-trained MobileNetV3 Large model for direction classification
    model_path = config.direction_model
    model = tf.keras.models.load_model(model_path)

    # Resize the image to the input size expected by MobileNetV3 Large (224x224) and preprocess
    img = cv2.resize(cattleImg, (224, 224))
    img = preprocess_input(img)

    # Perform inference
    prediction = model.predict(np.array([img]))
    class_index = np.argmax(prediction)

    # Compute global direction
    gd = (class_index * (360/config.DC) + yaw) % 360

    return gd

# Function to process each image and its corresponding XML file with bounding boxes
def process_image(folder, filename):

    # List to store attribute vectors for each cattle
    attribute_vectors = []

    # Read XML file containing cattle bounding boxes
    xml_path = os.path.join(folder, os.path.splitext(filename)[0] + '.xml')
    if not os.path.exists(xml_path):
        return attribute_vectors

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Read image
    img_path = os.path.join(folder, filename)
    img = cv2.imread(img_path)
    projections, yaw, datetime_taken = projection.compute_projections(img_path)  # Compute projections of the image (corners coordinates)
    (h, w) = img.shape[:2]

    # Iterate through objects in XML
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name != 'cattle':  # only objects classified as "cattle" will be counted
            continue

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Crop image to get cattle
        cattle_img = img[ymin:ymax, xmin:xmax]

        # Get global coordinates of the cattle
        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2
        global_coords = projection.local2global(projections, x_center/w, y_center/h)

        # Get attributes using classifiers
        color_class, color_score = getColor(cattle_img)
        state_class, state_score = getState(cattle_img)
        global_direction = getGlobalDirection(cattle_img, yaw)

        # Create attribute vector
        attribute_vector = [(global_coords[0], global_coords[1]), (color_class, color_score), (state_class, state_score), global_direction, datetime_taken]

        # Append attribute vector to list
        attribute_vectors.append(attribute_vector)

        # Save attribute vector to file
        if config.save_cattle_attributes_to_file:
            vec_path = os.path.join(folder, os.path.splitext(filename)[0] + '.vec')
            with open(vec_path, 'a') as f:
                f.write(','.join(str(attr) for attr in attribute_vector) + '\n')

    return attribute_vectors

# Method to counting all cattle from all images in a folder
def cattle_counting(folder):
    # List to store all counted cattle
    lt = []
    # Iterate through images in folder
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            li = process_image(folder, filename) # Process image and get all cattle features from it

            if len(li) == 0: # Image have no cattle
                continue
            elif len(lt) == 0: # None cattle had be counted yet, so, add all cattle to the total list
                lt.extend(li)
            # There are cattle in both lists. Need to check if there are duplicate cattle to do not count twice
            else:
                edges = computeCattleEdges(lt, li)

                # Check if there are candidate cattle duplicated
                if len(edges) > 0:
                    lt_index = list(np.arange(0, len(lt)))
                    li_index = list(np.arange(len(lt), len(lt) + len(li)))
                    maxFlow, chosenEdges = solveDuplicateCattle(lt_index, li_index, edges) # Perform maximum flow algorithm to found duplicated cattle

                    # Update duplicate cattle replacing the old features to the new one founded
                    for edge in chosenEdges:
                        lt[edge[0]] = li[edge[1] - len(lt)]
                    
                    # All cattle not duplicated are appended in the total list
                    for i, id in enumerate(li_index):
                        if id not in {edge[1] for edge in chosenEdges}:
                            lt.append(li[i])
                    
    if config.print_cattle_coordinate:
        print("\n\nList of cattle attributes:\n")
        print(lt)

    # lt contains the final attributes of all cattle counted in the folder (duplicate removed)
    print("\n\nTOTAL COUNTING: %d" %(len(lt)))
    return lt
            

# Call the cattle counting method
if __name__ == "__main__":
    folder = "playground"
    cattle_counting(folder)
