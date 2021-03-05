try:
    import tensorflow as tf
except ImportError as error:
    message = ('Impossible to import TensorFlow.\n'
               'To use TensorLy with the TensorFlow backend, '
               'you must first install TensorFlow!')
    raise ImportError(message) from error

import numpy as np

from . import Backend


class TensorflowBackend(Backend):
    backend_name = 'tensorflow'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=np.float32, device=None, device_id=None):
        if isinstance(data, tf.Tensor):
            return data

        out = tf.Variable(data, dtype=dtype)
        return out.gpu(device_id) if device == 'gpu' else out

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, tf.Tensor) or isinstance(tensor, tf.Variable)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        elif isinstance(tensor, tf.Variable):
            return tf.convert_to_tensor(tensor).numpy()
        else:
            return tensor

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return tf.experimental.numpy.clip(tensor, a_min, a_max)

    @staticmethod
    def sort(tensor, axis, descending = False):
        if descending:
            direction = 'DESCENDING'
        else:
            direction = 'ASCENDING'
            
        if axis is None:
            tensor = tf.reshape(tensor, [-1])
            axis = -1

        return tf.sort(tensor, axis=axis, direction = direction)
    
    def svd(self, matrix, full_matrices):
        """ Correct for the atypical return order of tf.linalg.svd. """
        S, U, V = tf.linalg.svd(matrix, full_matrices=full_matrices)
        return U, S, tf.transpose(a=V)
    
    def index_update(self, tensor, indices, values):
        if not isinstance(tensor, tf.Variable):
            tensor = tf.Variable(tensor)
            to_tensor = True
        else:
            to_tensor = False
        
        if isinstance(values, int):
            values = tf.constant(np.ones(self.shape(tensor[indices]))*values,
                                 **self.context(tensor))
        
        tensor = tensor[indices].assign(values)

        if to_tensor:
            return tf.convert_to_tensor(tensor)
        else:
            return tensor


for name in ['int64', 'int32', 'float64', 'float32', 'reshape', 'moveaxis', 'ndim',
             'where', 'copy', 'transpose', 'arange', 'ones', 'zeros', 'flip',
             'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min',
             'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
             'argmax', 'stack', 'conj', 'diag', 'einsum', 'shape', 'dot']:
    TensorflowBackend.register_method(name, getattr(tf.experimental.numpy, name))

for name in ['qr', 'eigh', 'solve']:
    TensorflowBackend.register_method(name, getattr(tf.linalg, name))
