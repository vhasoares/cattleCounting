Cc = 0.7692 # learned threshold for color softmax score
Cs = 0.8571 # learned threshold for state softmax score

DC = 10.0  # Number of possible classes in the direction model
DT = 15.0 # Maximum distance threshold
minR = 0.9 / DT  

# Turn ON/OFF each cattle attribute on counting method
color = False
state = False
velocity = True
direction = False


# Models path
color_model = 'models/getColorV3/v3-large_224_1.model'
state_model = 'models/getStateV3/v3-large_224_1.model'
direction_model = 'models/getDirectionV3/v3-large_224_1.model'
velocity_model = 'models/logistic_regression_model.pkl'

# General configuration
save_cattle_attributes_to_file = False  # To save the cattle attributes for each image in file
print_cattle_coordinate = False # Print all the cattle coordinate at the end of counting
plot_histogram_when_training = False