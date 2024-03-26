import math
import config
from datetime import datetime
from logisticRegression import CattleClassifier


# Given two coordinates, compute the direction in degrees from coordA to coordB
def calculate_direction(coordA, coordB):
    latA, lonA = coordA
    latB, lonB = coordB

    # Calculate the direction in degrees (0 degrees is north, angles increase clockwise)
    direction = math.degrees(math.atan2(latB - latA, lonB - lonA))

    # Convert the direction to the convention where north is 0 degrees and angles increase clockwise
    direction = 90 - direction  # Adjust for north being 0 degrees
    if direction < 0:
        direction += 360  # Ensure the result is positive

    return direction



# Given two coordinates, compute the distance in meters
def compute_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Radius of the Earth in meters
    R = 6371000.0

    # Compute the change in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate the distance between the coordinates
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


# Given the distance and the timestamp of two photos, compute the velocity
def compute_velocity(distance, timeA, timeB):

    # Compute the time difference in seconds
    time_diff = (timeB - timeA).total_seconds()

    # Avoid division by zero
    if time_diff == 0:
        return 0.0

    # Compute the velocity (distance / time) in m/s
    velocity_m_s = distance / time_diff

    # Convert velocity to km/h
    velocity_kmh = velocity_m_s * 3.6
    return velocity_kmh


# Compute the weight edges between the two lists of cattle (cumulative total and current image) based on attributes consensus
def computeCattleEdges(listTotal, listImage):
    E = []

    for listT_idx, cattleA in enumerate(listTotal):
        for listI_idx, cattleB in enumerate(listImage):

            # Verify Color attributes agreement
            color_classA = cattleA[1][0]  # Color class index of cattleA
            color_classB = cattleB[1][0]  # Color class index of cattleB
            color_softmax_scoreA = cattleA[1][1]  # Color Softmax score of cattleA
            color_softmax_scoreB = cattleB[1][1]  # Color Softmax score of cattleB

            # Ignore if the color of the cattle are different and both softmax score are greater than the threshold
            if color_classA != color_classB and color_softmax_scoreA > config.Cc and color_softmax_scoreB > config.Cc:
                continue

            # Verify State attributes agreement
            state_classA = cattleA[2][0]  # State class index of cattleA
            state_classB = cattleB[2][0]  # State class index of cattleB
            state_softmax_scoreA = cattleA[2][1]  # State Softmax score of cattleA
            state_softmax_scoreB = cattleB[2][1]  # State Softmax score of cattleB

             # Ignore if the state of the cattle are different and both softmax score are greater than the threshold
            if state_classA != state_classB and state_softmax_scoreA > config.Cs and state_softmax_scoreB > config.Cs:
                continue

            # Compute distance between objA and objB
            latA, lonA = cattleA[0]  # Global coordinates of cattleA
            latB, lonB = cattleB[0]  # Global coordinates of cattleB
            distance = compute_distance(latA, lonA, latB, lonB)

            tetaXY = 1.0
            # Verify if direction must be considered
            if config.direction:
            # Compute direction from cattleA to cattleB
                movementDirection = calculate_direction(cattleA[0], cattleB[0])
                classDirection = cattleA[3]
                directionDeviation = min(abs(classDirection - movementDirection), 360-abs(classDirection - movementDirection))
                tetaXY = 1.0 - (math.floor((directionDeviation/(360.0/config.DC))+0.5) * ((1.0-config.minR) / (config.DC/2.0)))


            # Ignore if the distance from cattleA to cattleB is greater than DT (distance threshold)
            if distance > (config.DT * tetaXY):
                continue

            probability = 1.0
            # Verify if velocity must be considered
            if config.velocity:
                # Extract the datetime_taken from each object
                timeA = datetime.strptime(cattleA[4], "%Y:%m:%d %H:%M:%S")  # Convert string to datetime
                timeB = datetime.strptime(cattleB[4], "%Y:%m:%d %H:%M:%S")  # Convert string to datetime

                # Compute the velocity between objA and objB
                velocity = compute_velocity(distance, timeA, timeB)

                classifier = CattleClassifier()
                # Predict probability for a velocity
                probability = classifier.predict(velocity)

            # If there is a positive probability of the cattle be the same, create and add the edge
            if probability > 0:
                # Add edge to the edge list E
                E.append((listT_idx, len(listTotal) + listI_idx, probability))

    return E




