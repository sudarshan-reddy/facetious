from os.path import join
import numpy as np
from IPython.display import Image, display
from resnet152.resnet152 import resnet152_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import SGD

class predictor:
    def __init__(self):
        img_rows, img_cols = 224, 224 # Resolution of inputs
        self.img_rows = img_rows
        self.img_cols = img_cols
        channel = 3
        batch_size = 8
        nb_epoch = 10
        
        weights_path = 'resnet152_weights_tf.h5'
        
        # Test pretrained model
        self.model = resnet152_model(img_rows, img_cols, channel, weights_path, load_top=False, new_top=True)
        sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    def read_and_prep_image(self, img_paths, img_height=224,
                             img_width=224):
        imgs = [load_img(img_path, target_size=(img_height, img_width)) for
                img_path in img_paths]
        print(imgs)
        img_array = np.array([img_to_array(img) for img in imgs])
        return preprocess_input(img_array)

    def predict(self, img_array):
        predictions = self.model.predict(img_array) 
        return predictions


if __name__ == '__main__':
    predict = predictor()
    directory = 'images'
    path = [join(directory, filename) for filename in ['867751.jpg',
                                                       'IMG_20180526_194347.jpg']]
    img_array = predict.read_and_prep_image(path)
    prediction = predict.predict(img_array)
    print(prediction)

