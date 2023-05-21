from keras.models import load_model
model_path = load_model("vehicle_CNN.h5")
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.image_utils import load_img, img_to_array


import os
#import matplotlib.pyplot as plt

data = []

for filename in os.listdir(r"C:\Users\dangc\OneDrive\Pictures\test_vehicle"):
    if filename.endswith("jpg"): 
        # Your code comes here such as 
        #print(r"C:\Users\dangc\OneDrive\Pictures\vehicle\" +filename)
        data.append(r"C:\Users\dangc\OneDrive\Pictures\test_vehicle"+'\\'+filename)
        
#print(data)

import numpy as np
import matplotlib.pyplot as plt
from keras.utils.image_utils import load_img, img_to_array

fig, axs = plt.subplots(nrows=5, ncols=12, figsize=(10,10))

# flatten the axis into a 1-d array to make it easier to access each axes
axs = axs.flatten()

fig.suptitle('Vehicle Detection',fontsize = 18)
for i in range(60):
    name =0
    img = load_img(data[i],target_size= (100,100))
    img1 = img_to_array(img)
    img1 = img1.reshape(1,100,100,3)
    img1 = img1.astype('float32')
    img1 = img1/255
    vehicle = np.argmax(model_path.predict(img1), axis = -1)
    if (vehicle == 1) :
       
       name = 'CAR'
       print(name) 
    if (vehicle == 2) :
       name ='TRUCK'
       print(name)
    axs[i].imshow(img)
    axs[i].set(title= name)
    #plt.imshow(img)
plt.tight_layout()
plt.show()

