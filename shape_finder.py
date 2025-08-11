import tensorflow as tf

TFLITE_PATH = "./whisper-tiny.en.tflite"
interpreter = tf.compat.v1.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()


print("Input shape :\n", interpreter.get_input_details())
print("Output shape: \n", interpreter.get_output_details())
