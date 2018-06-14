from keras.applications import inception_v3
from keras.layers import Dense, Flatten, Dropout
from keras.models import  Model
from keras.optimizers import SGD
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

X_data = np.load('X_data.npy')
Y_data = np.load('Y_data.npy')


Base = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
X_ = Base.output
X_ = Dense(512, activation='relu')(X_)
predictions = Dense(2, activation='softmax')(X_)
model = Model(inputs=Base.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
callb = ModelCheckpoint(filepath='model.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

model.fit(x=X_data,y=Y_data,batch_size=20,epochs=10,validation_split=0.15,shuffle=True,callbacks=[callb])


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
