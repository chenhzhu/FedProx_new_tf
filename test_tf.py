
import tensorflow as tf
from tensorflow.python.client import device_lib


print(tf.__version__)
tf.config.list_physical_devices('GPU')
print(device_lib.list_local_devices())
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)

local_device_protos = device_lib.list_local_devices()
# print(local_device_protos)
 
# Only print GPU devices.
[print(x) for x in local_device_protos if x.device_type == 'GPU']