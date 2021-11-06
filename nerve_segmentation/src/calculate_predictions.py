import tflite_runtime.interpreter as tflite
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import time

#Create a data directory and copy the imgs_test.npy in that directory.
img_test_path = os.path.abspath('../data/imgs_test.npy')

def load_test_data():
    print(f"Loading test data from : {img_test_path}")
    imgs_test = np.load(img_test_path)
    return imgs_test

IMG_ROWS, IMG_COLS = 80, 112
img_rows = IMG_ROWS
img_cols = IMG_COLS

def preprocess(imgs, to_rows=None, to_cols=None):
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols

    imgs_p = np.ndarray((imgs.shape[0], to_rows, to_cols, imgs.shape[3]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, :, :, 0] = resize(imgs[i, :, :, 0], (to_rows, to_cols), preserve_range=True)

    return imgs_p

WEIGHT_PATH = os.path.abspath('../weights/QAT_INT8_Nerve_Segmentation_edgetpu.tflite')
interpreter = tflite.Interpreter(model_path = WEIGHT_PATH, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
input_details = interpreter.get_input_details()
print(f"Input Details : {input_details}")
output_details = interpreter.get_output_details()
print(f"Output Details : {output_details}")

imgs_test = load_test_data()
imgs_test = preprocess(imgs_test)
imgs_test = imgs_test.astype('float32')

MEAN = 98.06 #Calculated from train data.
STD = 51.57 #Calculated from train data.

imgs_test -= MEAN
imgs_test /= STD

print(f"Shape of Inages: {imgs_test.shape}")

#Doing batch prediction with tflite model.

interpreter.resize_tensor_input(input_details[0]['index'],(5508, 80, 112, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (5508, 80,112,1))
interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'], imgs_test)
print(f"Performing batch prediction...")
start_time = time.perf_counter()
interpreter.invoke()
end_time = time.perf_counter()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(f"Batch prediction over...")
print(f"Total Time : {end_time - start_time} sec.")

print(f"Saving Results...")
np.save('QAT_INT8_Predictions.npy', tflite_results)