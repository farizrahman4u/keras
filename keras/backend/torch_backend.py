import numpy as np
import torch
import torch.nn.functional as F
import keras
from .graph import *
from collections import defaultdict
from contextlib import contextmanager
from .common import set_image_dim_ordering, image_dim_ordering
from .common import floatx, epsilon, image_data_format


py_all = all
py_sum = sum



_LEARNING_PHASE = Tensor(name='keras_learning_phase')
_UID_PREFIXES = defaultdict(int)


def learning_phase():
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    _LEARNING_PHASE = value


def get_uid(prefix=''):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    global _UID_PREFIXES
    _UID_PREFIXES = defaultdict(int)


NAME_SCOPE_STACK = []


@contextmanager
def name_scope(name):
    global NAME_SCOPE_STACK
    NAME_SCOPE_STACK.append(name)
    yield
    NAME_SCOPE_STACK.pop()


def _prepare_name(name, default):
    prefix = '/'.join(NAME_SCOPE_STACK)
    if name is None:
        return prefix + '/' + default
    return prefix + '/' + name


def is_keras_tensor(x):
    return hasattr(x, '_keras_history')



def _is_num(x):
    try:
        float(x)
        return True
    except:
        return 'numpy' in str(type(x))


def _get_shape(x):
    if hasattr(x, 'value'):
        return x.value.size()
    if hasattr(x, 'shape'):
        return x.shape
    if hasattr(x, 'size'):
        return tuple(x.size())
    if _is_num(x):
        return ()
    return None


def make_keras_tensor(tensor, uses_learning_phase=False):
    tensor._keras_shape = int_shape(tensor)
    tensor._uses_learning_phase = uses_learning_phase


def variable(value, dtype=None, name=None, constraint=None):
    if isinstance(value, Tensor):
        value = value.value
    if isinstance(value, torch.autograd.Variable):
        value = value.data
    if 'torch' in str(type(value)):
        value = value.numpy()
    if dtype is None:
        dtype = keras.backend.floatx()
    if value.dtype != dtype:
        value = np.cast[dtype](value)
    torch_tensor = torch.from_numpy(value)
    torch_variable = torch.autograd.Variable(torch_tensor, requires_grad=True)
    return torch_variable
    ktorch_variable = Variable(torch_variable, name=name)
    ktorch_variable.constraint = None
    make_keras_tensor(ktorch_variable)
    return ktorch_variable


def constant(value, dtype=None, shape=None, name=None):
    value = np.array(value)
    if dtype is None:
        dtype = keras.backend.floatx()
    if value.dtype != dtype:
        value = np.cast[dtype](value)
    if value.shape == ():
        if shape is None:
            shape = ()
        value = np.ones(shape) * value
    torch_tensor = torch.from_numpy(value)
    torch_variable = torch.autograd.Variable(torch_tensor, requires_grad=False)
    return torch_variable
    ktorch_variable = Variable(torch_variable, name=name)
    make_keras_tensor(ktorch_variable)
    return ktorch_variable


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    ph = SymbolicTensor()
    ph._ktorch_placeholder = True
    return ph
    if sparse:
        raise Exception('Sparse tensors are not supported yet :( ')
    if dtype is None:
        dtype = keras.backend.floatx()
    ktorch_tensor = Tensor(name=name, shape=shape, ndim=ndim, dtype=dtype)
    make_keras_tensor(ktorch_tensor)
    ktorch_tensor._ktorch_placeholder = True
    return ktorch_tensor


def is_placeholder(x):
    """Returns whether `x` is a placeholder.

    # Arguments
        x: A candidate placeholder.

    # Returns
        Boolean.
    """
    return hasattr(x, '_ktorch_placeholder') and x._ktorch_placeholder


def shape(x):
    if hasattr(x, 'value'):
        return Variable(tuple(x.value.size()))
    elif hasattr(x, 'shape'):
        return Variable(x.shape)
    else:
        raise Exception('Tensor shape not available.')


def int_shape(x):
    if hasattr(x, 'value'):
        return tuple(x.value.size())
    elif hasattr(x, 'shape'):
        return x.shape
    else:
        raise Exception('Tensor shape not available.')


def ndim(x):
    x_shape = _get_shape(x)
    if x_shape is None:
        return None
    else:
        return len(x_shape)


def dtype(x):
    if isinstance(x, Tensor):
        x = x.eval()
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    return type(x)


def eval(x):
    y = x.eval()
    if 'torch' in str(type(x)) and hasattr(y, 'data'):
        y = y.data
    if hasattr(y, 'numpy'):
        y = y.numpy()
    return y


def zeros(shape, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.ones(shape), dtype, name)


def eye(size, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.eye(size), dtype, name)


def ones_like(x, dtype=None, name=None):
    return x * 0 + 1


def zeros_like(x, dtype=None, name=None):
    return x * 0


def identity(x):
    return x + 0


def count_params(x):
    return np.prod(x.eval().size())


def random_uniform_variable(shape, low, high, dtype=None, name=None):
    return variable(np.random.uniform(low=low, high=high, size=shape),
                    dtype=dtype, name=name)


def random_normal_variable(shape, mean, scale, dtype=None, name=None):
    return variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                    dtype=dtype, name=name)


def cast(x, dtype):
    return x.type(dtype)

# UPDATES OPS


def update(x, new_x):
    return (x, new_x)


def update_add(x, increment):
    return (x, x + increment)


def update_sub(x, decrement):
    return (x, x - decrement)


def moving_average_update(variable, value, momentum):
    return (variable, variable * momentum + value * (1. - momentum))

def bias_add(x, bias, data_format=None):
    def _bias_add(X, data_format):
        x, bias = X
        from keras.backend import image_data_format, ndim, reshape
        if data_format is None:
            data_format = image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format ' + str(data_format))
        if ndim(bias) != 1 and ndim(bias) != ndim(x) - 1:
            raise ValueError('Unexpected bias dimensions %d, '
                             'expect to be 1 or %d dimensions'
                             % (ndim(bias), ndim(x) - 1))
        bias_shape = tuple(bias.size())
        ndim_x = len(x.size())
        ndim_bias = len(bias_shape)
        if ndim_x == 5:
            if data_format == 'channels_first':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, bias_shape[0], 1, 1, 1))
                else:
                    bias = reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
            elif data_format == 'channels_last':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, 1, 1, 1, bias_shape[0]))
                else:
                    bias = reshape(bias, (1,) + bias_shape)
        elif ndim_x == 4:
            if data_format == 'channels_first':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, bias_shape[0], 1, 1))
                else:
                    bias = reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
            elif data_format == 'channels_last':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, 1, 1, bias_shape[0]))
                else:
                    bias = reshape(bias, (1,) + bias_shape)
        elif ndim_x == 3:
            if data_format == 'channels_first':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, bias_shape[0], 1))
                else:
                    bias = reshape(bias, (1, bias_shape[1], bias_shape[0]))
            elif data_format == 'channels_last':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, 1, bias_shape[0]))
                else:
                    bias = reshape(bias, (1,) + bias_shape)
        return x.add(bias.expand_as(x))
    return _bias_add([x, bias], data_format)


def dot(x, y):
    def _dot(X):
        x, y = X
        x_ndim = ndim(x)
        y_ndim = ndim(y)
        if x_ndim == 2 and y_ndim == 2:
            return torch.mm(x, y)
        if x_ndim == 2 and y_ndim == 1:
            return torch.mv(x, y)
        if x_ndim == 1 and y_ndim == 2:
            return torch.mv(y, x)
        if x_ndim == 1 and y_ndim == 1:
            return torch.dot(x, y)
        else:
            raise Exception('Unsupported tensor ranks for dot operation : ' + str(x_ndim) + ' and ' + str(y_ndim) + '. Inputs : ' +
                str(x) + ', ' + str(y))
    return _dot([x, y])


def batch_dot(x, y, axes=None):
    if type(axes) is int:
        axes = (axes, axes)
    def _dot(X):
        x, y = X
        x_shape = x.size()
        y_shape = y.size()
        x_ndim = len(x_shape)
        y_ndim = len(y_shape)
        if x_ndim <= 3 and y_ndim <= 3:
            if x_ndim < 3:
                x_diff = 3 - x_ndim
                for i in range(diff):
                    x = torch.unsqueeze(x, x_ndim + i)
            else:
                x_diff = 0
            if y_ndim < 3:
                y_diff = 3 - y_ndim
                for i in range(diff):
                    y = torch.unsqueeze(y, y_ndim + i)
            else:
                y_diff = 0
            if axes[0] == 1:
                x = torch.transpose(x, 1, 2)
            elif axes[0] == 2:
                pass
            else:
                raise Exception('Invalid axis : ' + str(axes[0]))
            if axes[1] == 2:
                x = torch.transpose(x, 1, 2)
            # -------TODO--------------#


def transpose(x):
    dim_order = list(reversed(range(ndim(x))))
    return torch.Tensor.permute(x, *dim_order)


# ELEMENT-WISE OPERATION

def max(x, axis=None, keepdims=False):
    return torch.max(axis, keepdim=keepdims)[0]


def min(x, axis=None, keepdims=False):
    y = torch.min(x, axis, keepdim=keepdims)[0]


def sum(x, axis=None, keepdims=False):
    if type(axis) in (tuple, list):
        if keepdims:
            for i in range(len(axis)):
                x = torch.sum(x, axis[i], keepdim=True)
            return x
        else:
            for i in range(len(axis)):
                x = torch.sum(x, axis[i] - i)
            return x
    else:
        return torch.sum(x, axis=axis, keepdim=keepdims)


def prod(x, axis=None, keepdims=False):
    return torch.prod(x, axis, keepdim=keepdims)


def std(x, axis=None, keepdims=False):
    return torch.std(x, axis, keepdim=keepdims)

def var(x, axis=None, keepdims=False):
    return torch.var(x, axis, keepdim=keepdims)

def cumsum(x, axis=0):
    return torch.cumsum(x, axis)
#~~~~~~~~~~~~~~ UNIMPLEMENTED IN PYTORCH !! ~~~~~~~~~~~~~~#


def cumprod(x, axis=0):
    pass
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def mean(x, axis=None, keepdims=False):
    return torch.mean(x, axis, keepdim=keepdims)


def any(x, axis=None, keepdims=False):
    y = torch.sum(x != 0, axis) != 0
    return y if keepdims else torch.squeeze(y, axis)


def all(x, axis=None, keepdims=False):
    y = torch.sum(x == False, axis) == 0
    return y if keepdims else torch.squeeze(y, axis)


def argmax(x, axis=-1, keepdims=False):
    return torch.max(x, axis, keepdim=keepdims)[1]

def argmin(x, axis=-1, keepdims=False):
    return torch.min(x, axis, keepdim=keepdims)[1]


def square(x):
    return x ** 2

def abs(x):
    return torch.abs(a)


def sqrt(x):
    return torch.sqrt(x)


def exp(x):
    return torch.exp(x)


def log(x):
    return torch.log(x)


def logsumexp(x, axis=None, keepdims=False):
    return torch.log(torch.sum(torch.exp(x), axis, keepdim=keepdims))

def round(x):
    return torch.round(x)


def sign(x):
    return torch.sign(x)


def pow(x, exp):
    return torch.pow(x, exp)


def clip(x, min_value, max_value):
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    return torch.clamp(x, min_value, max_value)


def equal(x, y):
    return x == y

def not_equal(x, y):
    return x != y

def greater(x, y):
    return x > y

def greater_equal(x, y):
    return x >= y

def less(x, y):
    return x < y

def less_equal(x, y):
    return x <= y


def maximum(x, y):
    return torch.max(x, y)


def minimum(x, y):
    return torch.min(x, y)


def sin(x):
    return torch.sin(x)

def cos(x):
    return torch.cos(x)


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    return torch.cat(tensors, axis)


def reshape(x, shape):
    return x.view(shape)

def permute_dimensions(x, pattern):
    return x.permute(*pattern)


def arange(start, stop=None, step=1, dtype='int32'):
        #TODO : Other datatypes
        return torch.arange(start, stop, step).int()


def flatten(x):
    return x.view([-1])


def expand_dims(x, axis=-1):
    return torch.unsqueeze(x, axis)


def squeeze(x, axis):
    return torch.squeeze(x, axis)


def stack(x, axis=0):
    return torch.stack(x, axis)

def one_hot(indices, num_classes):
    # Not differentiable
    temp = indices.view(-1,1).long().data
    batch_size = temp.size()[0]
    y = torch.zeros(batch_size, num_classes)
    return y.scatter_(1, temp, 1)


# VALUE MANIPULATION

def get_value(x):
    return x.eval().data.numpy()


def batch_get_value(ops):
    return [x.eval().data.numpy() for x in ops]


def set_value(x, value):
    value = np.asarray(value)
    x.value.data = torch.from_numpy(value)


def batch_set_value(tuples):
    for x, value in tuples:
        set_value(x, value)


def get_variable_shape(x):
    return tuple(x.value.size())


def print_tensor(x, message=''):
    print(message, x.value.data)


## NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    if alpha != 0.:
        negative_part = F.relu(-x)
    x = F.relu(x)

    if max_value is not None:
        print ("Meh")
        x = torch.clamp(x, max=max_value)

    if alpha != 0:
        x -= alpha * negative_part
    return x


def elu(x, alpha=1.):
    return F.elu(x)


def softmax(x):
    return F.softmax(x)


def softplus(x):
    return F.softplus(x)


def softsign(x):
    return F.softsign(x)


def sigmoid(x):
    return F.sigmoid(x)


def hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    return torch.clamp(x, 0., 1.)


def tanh(x):
    return F.tanh(x)

def dropout(x, level, noise_shape=None, seed=None):
    # No support for noise shape and seed as of now
    return F.dropout(x, p=level, training=True)


def l2_normalize(x, axis):
    return torch.nn.functional.normalize(x, p=2, dim=axis)


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    #TODO dtype
    #TODO seed
    return torch.from_numpy(np.random.normal(mean, stddev, shape))


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    #TODO dtype
    #TODO seed
    return torch.from_numpy(np.random.uniform(minval, maxval, shape))


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    #TODO dtype
    #TODO seed
    return torch.from_numpy(np.random.binomial(1, p, shape))


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    x = random_normal(shape, mean, stddev, dtype, seed)
    return torch.clamp(x, mean - 2 * stddev, mean + 2 * stddev)

def tile(x, n):
    if type(n) is int:
        n = [1] * (len(x.size()) - 1) + [n]
    return x.repeat(*n)

def rnn(step_function, inputs, initial_states, go_backwards=False,
        mask=None, constants=None, unroll=False, input_length=None):
    outputs = []
    states = initial_states[:]
    timesteps = inputs.size(1)
    if constants is None:
        constants = []
    if mask is None:
        for t in range(timesteps):
            input_t = inputs[:, t]
            output_t, states = step_function(input_t, states + constants)
            outputs.append(output_t)
        return outputs[-1], outputs, states
    else:
        sample = step_function(inputs[:, 0], initial_states + constants)
        outputs.append(sample * 0)
        for t in range(timesteps):
            input_t = inputs[:, t]
            mask_t = mask[:, t]
            output_t, new_states = step_function(input_t, states + constants)
            output_t = mask_t * output_t + (1 - mask) * outputs[-1]
            outputs.append(output_t)
            for i in range(len(states)):
                states[i] = mask_t * new_states[i] + (1 - mask_t) * states[i]
        return outputs[-1], outputs[1:], states