import tensorflow as tf


def detect_data_format():
    return 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'

def is_channels_first(data_format):
    data_format = data_format.lower()
    return data_format in ('channels_first', 'nchw')
    
def to_data_format(tensor, current_data_format, target_data_format):
    if current_data_format == target_data_format:
        return tensor
    
    if is_channels_first(target_data_format):
        return tf.transpose(tensor, perm=[0, 3, 1, 2])
    
    return tf.transpose(tensor, perm=[0, 2, 3, 1])

def channel_dim(data_format):
    if is_channels_first(data_format):
        return 1
    return 3

def wh_dims(data_format):
    if is_channels_first(data_format):
        return (2, 3)
    return (1, 2)

def channel_axis(f, data_format, *args, **kwargs):
    return f(*args, axis=channel_dim(data_format), **kwargs)

def _shape(shape, dim):
    if isinstance(dim, (list, tuple)):
        return shape[dim[0]:dim[1]+1]
    return shape[dim]

def static_shape(tensor, dims):
    return _shape(tensor.shape.as_list(), dims)

def tensor_shape(tensor, dims):
    return _shape(tf.shape(tensor), dims)

def image_resize(inputs, size, data_format, method=0):
    inputs = to_data_format(inputs, data_format, 'channels_last')        
    inputs = tf.image.resize_images(inputs, size, method=method)
    inputs = to_data_format(inputs, 'channels_last', data_format)
    return inputs
