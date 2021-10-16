import tensorflow as tf

class ModelOptimization:
    def __init__(self):
        self._weight_file_path = None
        self._representative_dataset = None
        self._model = None
        
    @property
    def weight_file_path(self):
        return self._weight_file_path

    @weight_file_path.setter
    def weight_file_path(self, new_weight_file_path):
        self._weight_file_path = new_weight_file_path

    @property
    def representative_dataset(self):
        return self._representative_dataset

    @representative_dataset.setter
    def representative_dataset(self, new_dataset):
        self._representative_dataset = new_dataset

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model

    def int_8_quantization(self):
        print("Performing INT-8 Quantization...")
        model = self.model.load_weights(self.weight_file_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_model_int_8_quant = converter.convert()
        with open('INT8_quantized.tflite', 'wb') as fp_int8_file:
            fp_int8_file.write(tflite_model_int_8_quant)

