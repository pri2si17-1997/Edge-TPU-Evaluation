# Run this code on coral dev board.

try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow as tf

import pandas as pd
import cv2
import time
import numpy as np
import os
import glob2
import json

#Change Image path according to your path.
IMAGE_PATH = os.path.abspath('/SDcard/work/test_images/')

#Change this while evaluating other models.
WEIGHT_PATH = os.path.abspath('../weights/QAT_INT8_edgetpu.tflite')

images = sorted(glob2.glob(IMAGE_PATH + '/*.png'))
print(f"Total number of Images : {len(images)}")

def evaluate_timing():
    total_time = 0
    try:
        interpreter = tflite.Interpreter(model_path = WEIGHT_PATH,experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    except Exception as ex:
        interpreter = tf.lite.Interpreter(model_path = WEIGHT_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    prediction_op = dict()
    
    for _, img_item in enumerate(images):
        img = cv2.imread(img_item)
        img = cv2.resize(img, (224, 224)) / 255
        test_image = np.expand_dims(img, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details['index'], test_image)
        start_time = time.perf_counter()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_op[img_item] = output_data.tolist()
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        total_time += time_taken

    print(f"Total Time taken : {total_time} seconds.")
    print(f"Average Time Taken : {total_time / len(images)} seconds")

    #Save the Predictions in File.
    with open('prediction.json', 'w') as op:
        json.dump(prediction_op, op)

if __name__ == "__main__":
    evaluate_timing()