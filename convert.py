import scipy.io as sio
import numpy as np
from PIL import Image

# Loading data
test = sio.loadmat('anno.mat')

# Declaring Variables
new_width  = 96
new_height = 96

# Manipulating Images
X_data = []
for i in range(9344):
    current_image = Image.open('images/' + test['anno']['files'][0][0][0][i][0])
    cropped_image = current_image.crop((test['anno']['objbbs'][0][0][i][0],test['anno']['objbbs'][0][0][i][1],test['anno']['objbbs'][0][0][i][2],test['anno']['objbbs'][0][0][i][3]))
    resized_image = cropped_image.resize((new_width, new_height), Image.ANTIALIAS)
    image_array = np.asarray(resized_image)
    X_data.append(image_array)

# Saving image data into X_data
np.save('X_data.npy',X_data)


# Loading and saving output data
Y_array = np.asarray(test['anno']['y'][0][0])
Y_final = np.rot90(Y_array)
np.save('Y_data.npy', Y_final)