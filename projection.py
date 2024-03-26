from PIL import Image
from PIL.ExifTags import TAGS
from math import tan, radians, degrees, sin, cos, sqrt, atan, pi, asin, atan2

def sensor_sizeXY(focallen, focal35mm, imageW, imageH):
    #Function to compute the width and height of the camera sensor based on the len, crop factor and the image size
    
    fullFrameSize = 43.2666 #technical default value

    
    cropFactor = (focal35mm / focallen)
    
    #Sensor diagonal size
    sensorSize = fullFrameSize / cropFactor
    
    #Image diagonal size (hypotenuse)
    imageSize = sqrt(imageW**2 + imageH**2)
    
    #Ratio of image size to sensor size.
    ratio = imageSize / sensorSize
    
    #Applying the same ratio for the height and width of the image to get the size of the sensor.
    x_sensor = imageW / ratio
    y_sensor = imageH / ratio 
    
    return x_sensor, y_sensor


# Function that converts the points in the image as the location of the cattle
def local2global(projections, x, y):
	p1lat = float(projections[0])
	p1lng = float(projections[1])
	p2lat = float(projections[2])
	p2lng = float(projections[3])
	p3lat = float(projections[4])
	p3lng = float(projections[5])
	p4lat = float(projections[6])
	p4lng = float(projections[7])

	# point between the left vertices of the image
	latA = ((p4lat - p1lat) * y) + p1lat
	lngA = ((p4lng - p1lng) * y) + p1lng

	# point between the right vertices of the image
	latB = ((p3lat - p2lat) * y) + p2lat
	lngB = ((p3lng - p2lng) * y) + p2lng

	# point where the cattle is
	latitude = ((latB - latA) * x) + latA
	longitude = ((lngB - lngA) * x) + lngA

	return [latitude, longitude]



def new_coordinate(latitude, longitude, distance_meters, yaw_degree):
    """
        This function calculates a new latitude and longitude based on an original position, distance in meters, and yaw angle.

        Args:
            latitude: Original latitude in decimal degrees.
            longitude: Original longitude in decimal degrees.
            distance_meters: Distance to move in meters.
            yaw_deg: Yaw angle in degrees

        Returns:
            A list containing the new latitude and longitude (decimal degrees).
  """
    earth_radius = 6371000.0  # in meters
    angular_distance = distance_meters / earth_radius
    yaw_radians = radians(yaw_degree)
    lat_radians = radians(latitude)
    lon_radians = radians(longitude)

    new_latitude = asin(sin(lat_radians) * cos(angular_distance) +
                             cos(lat_radians) * sin(angular_distance) * cos(yaw_radians))

    new_longitude = lon_radians + atan2(sin(yaw_radians) * sin(angular_distance) * cos(lat_radians),
                                             cos(angular_distance) - sin(lat_radians) * sin(new_latitude))

    new_latitude = degrees(new_latitude)
    new_longitude = degrees(new_longitude)
    return [new_latitude, new_longitude]
    

def compute_projections(image):
    
    # Camera attributes to compute projections (if the image doesn't contain the information)

    # Camera sensor size in mm (the default values uses as Reference the DJI Mavic Pro camera)
    x_sensor = 6.25752 # width of sensor in mm
    y_sensor = 4.69314 # heigth of sensor in mm
    focallen = 4.7 # real focal lenght of the lens in mm
    
    altitude = 90.0 # Default value 
    
    yawDegree = 0.0 #Default value for yaw degree (north direction)
    
    imageW = 4000 # Image Width size
    imageH = 3000 # Image Height size 

    # Read the latitude and longetude; Image Size; FocalLength and FocalLenghtIn35mmFilm from EXIF data
    lat = 0.0
    lng = 0.0
    datetime_taken = None
    values = None
    for (k,v) in Image.open(image)._getexif().items():
        tag = TAGS.get(k)
        if tag == "GPSInfo":
            values = v  
        elif tag == "FocalLength":
            focallen = v[0]/v[1]
        elif tag == "FocalLengthIn35mmFilm":
            focal35mm = v
        elif tag == "ExifImageWidth":
            imageW = v
        elif tag == "ExifImageHeight":
            imageH = v
        elif tag == "DateTimeOriginal":
            datetime_taken = v
            
    x_sensor, y_sensor = sensor_sizeXY(focallen, focal35mm, imageW, imageH)
    
    # Convert the coordinates from degree to decimals
    lat = (values[2][0][0]/values[2][0][1]) + (values[2][1][0]/values[2][1][1])/60 + (values[2][2][0]/values[2][2][1])/3600
    if values[1] == 'S':
        lat = lat * -1
    lng = (values[4][0][0]/values[4][0][1]) + (values[4][1][0]/values[4][1][1])/60 + (values[4][2][0]/values[4][2][1])/3600
    if values[3] == 'W':
        lng = lng * -1

    
    # Read the altitude from XMP data
    with open(image, "rb") as fin:
        img = fin.read()
    imgAsString=str(img)
    xmp_start = imgAsString.find('<x:xmpmeta')
    xmp_end = imgAsString.find('</x:xmpmeta')
    if xmp_start != xmp_end:
        xmpString = imgAsString[xmp_start:xmp_end+12]
        x = xmpString.find('RelativeAltitude')
        altitude = float(xmpString[x::].split("\\n")[0].split("\"")[1])
        x = xmpString.find('FlightYawDegree')
        yawDegree = float(xmpString[x::].split("\\n")[0].split("\"")[1])
        x = xmpString.find('FlightPitchDegree')
        pitchDegree = float(xmpString[x::].split("\\n")[0].split("\"")[1])
        x = xmpString.find('GimbalPitchDegree')
        gimbalPitchDegree = float(xmpString[x::].split("\\n")[0].split("\"")[1])

    #print("\n\nFocallen: %.3f\nFocal35: %.3f\nImageW: %.3f\nImageH: %.3f\nxSensor: %.3f\nySensor: %.3f\naltitude: %.3f\nFlightPitch: %.3f\nGimbalPitch: %.3f\n\n" %(focallen, focal35mm, imageW, imageH, x_sensor, y_sensor, altitude, pitchDegree, gimbalPitchDegree))
    
    # Calculate field of view
    xview = 2.0*degrees(atan(x_sensor/(2.0*focallen)))
    yview = 2.0*degrees(atan(y_sensor/(2.0*focallen)))

    # Adjust the projection by some eventual pitch different than -90
    if gimbalPitchDegree is not None and gimbalPitchDegree != -90 and gimbalPitchDegree != 0:
        gimbalPitchDegree = -90 - gimbalPitchDegree
        pitchDegree = pitchDegree - gimbalPitchDegree
    
    distX = altitude*tan(radians(.5*xview)) # Distance in meters from drone to left and right borders of the image
    distY_top = altitude*tan(radians(.5*(yview-pitchDegree))) # Distance in meters from drone to top border of the image
    distY_botton = altitude*tan(radians(.5*(yview+pitchDegree))) # Distance in meters from drone to botton border of the image
    
    
    #-------------------------------
    #Computing rotation of the image
    #-------------------------------

    distDiagonal_top = sqrt(distX**2 + distY_top**2)
    distDiagonal_botton = sqrt(distX**2 + distY_botton**2)
    
    #Convert Yaw from (-180 +180) to (0  360) where 0 means the image points north
    if yawDegree < 0:
        yawDegree = 180.0 + (180.0 - abs(yawDegree))
    
    # Angles of the right triangle formed by the diagonal of each vertex of the image.
    ang1DiagTrian_top = degrees(atan(distX/distY_top))
    ang2DiagTrian_top = degrees(atan(distY_top/distX))
    ang1DiagTrian_botton = degrees(atan(distX/distY_botton))
    ang2DiagTrian_botton = degrees(atan(distY_botton/distX))
    
    # Yaw angle from each image diagonals
    yawTopRight = (yawDegree + ang1DiagTrian_top) % 360.0
    yawBottonRight = (yawTopRight + 2.0 * ang2DiagTrian_botton) % 360.0
    yawBottonLeft = (yawBottonRight + 2.0 * ang1DiagTrian_botton) % 360.0
    yawTopLeft = (yawBottonLeft + 2.0 * ang2DiagTrian_top) % 360.0
    

    projections = []

    projections.extend(new_coordinate(lat, lng, distDiagonal_top, yawTopLeft))  # Top left corner coordinate
    projections.extend(new_coordinate(lat, lng, distDiagonal_top, yawTopRight))  # Top right corner coordinate
    projections.extend(new_coordinate(lat, lng, distDiagonal_botton, yawBottonRight))  # Botton right corner coordinate
    projections.extend(new_coordinate(lat, lng, distDiagonal_botton, yawBottonLeft))   # Botton left corner coordinate


    return projections, yawDegree, datetime_taken
    

# Compute projection of some image
if __name__ == "__main__":

    projections, _, _ = compute_projections("playground/teste2.jpg")
    
    cont = 0
    for x in projections:
        if cont == 2:
            cont = 0
            print("")
        print("%.8f" % x)
        cont+=1
