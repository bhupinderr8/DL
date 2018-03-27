import scipy.io as sio
from PIL import Image


test = sio.loadmat('anno.mat')
X=0
Y=0
for i in range(9344):
    current_image = Image.open('images/' + test['anno']['files'][0][0][0][i][0])
    cropped_image = current_image.crop((test['anno']['objbbs'][0][0][i][0],test['anno']['objbbs'][0][0][i][1],test['anno']['objbbs'][0][0][i][2],test['anno']['objbbs'][0][0][i][3]))
    X = X + cropped_image.size[0]
    Y = Y + cropped_image.size[1]

print (X/9344)
print (Y/9344)