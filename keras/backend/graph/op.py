from .tensor import Tensor
from .node import Node
import numpy as np
import inspect

def _to_list(x):
    if type(x) is not list:
        return [x]
    return x

def _is_num(x):
    try:
        float(x)
        return True
    except:
        return 'numpy' in str(type(x))


def _get_shape(x):
    if hasattr(x, 'shape'):
        return x.shape
    if hasattr(x, 'value'):
        return x.value.size()
    if _is_num(x):
        return ()
    return None


class Op(object):

    def __init__(self, arguments={}):
        if not hasattr(self, 'num_inputs'):
            self.num_inputs = None
        self.arguments = arguments

    def call(self, x):
        # imperative code goes here
        return x

    def __call__(self, x):
        x_type = type(x)
        if x_type in [list, tuple]:
            x = list(x)
            tensors = [t for t in x if (t is not None) and not _is_num(t)]
            # search for non ktorch objs
            found = False
            for t in tensors:
                if not isinstance(t, Tensor):
                    found = True
                    print(t)
                    print(type(t))
                    break
            if found:
                for i in range(len(x)):
                    t = x[i]
                    if isinstance(t, Tensor):
                        x[i] = t.eval()
                is_symbolic = False
            else:
                is_symbolic = True
            x_len = len(x)
            if x_len == 1:
                x = x[0]
            else:
                x = list(x)
        else:
            x_len = 1
            is_symbolic = isinstance(x, Tensor)
        self._check_num_inputs(x_len)
        if not is_symbolic:
            if not self.arguments:
                return self.call(x)
            args_type = type(self.arguments)
            if args_type is dict:
                return self.call(x, **self.arguments)
            else:
                return self.call(x, *self.arguments)
        y = Tensor()
        y.op = self
        y.inputs = x
        shape_argspec = inspect.getargspec(self.compute_output_shape)
        dtype_argspec = inspect.getargspec(self.compute_output_dtype)
        if type(self.arguments) is dict:
            if len(shape_argspec.args) > 2 or shape_argspec.keywords is not None:
                y.shape = self.compute_output_shape(x, **self.arguments)
            else:
                y.shape = self.compute_output_shape(x)
            if len(dtype_argspec.args) > 2 or dtype_argspec.keywords is not None:
                y.dtype = self.compute_output_dtype(x, **self.arguments)
            else:
                y.dtype = self.compute_output_dtype(x)
        elif type(self.arguments) is list:
            if len(shape_argspec.args) > 2 or shape_argspec.varargs is not None:
                y.shape = self.compute_output_shape(x, *self.arguments)
            else:
                y.shape = self.compute_output_shape(x)
            if len(dtype_argspec.args) > 2 or dtype_argspec.varargs is not None:
                y.dtype = self.compute_output_dtype(x, *self.op.arguments)
            else:
                self.dtype = self.compute_output_dtype(x)
        else:
            self.shape = self.compute_output_shape(x)
            self.dtype = self.compute_output_dtype(x)
        Node(x, y)
        return y

    def _check_num_inputs(self, n):
        if not hasattr(self, 'num_inputs') or self.num_inputs is None:
            return
        class_name = self.__class__.__name__
        num_inputs = self.num_inputs
        if type(num_inputs) is str:
            if '+' in num_inputs:
                min_num_inputs = int(num_inputs[:-1])
                if n < min_num_inputs:
                    raise Exception(class_name + ' expected at least ' + str(min_num_inputs) + ' inputs but received only ' + str(n) + ' inputs.')
                return
            else:
                num_inputs = int(num_inputs)
        if num_inputs != n:
            raise Exception(class_name + ' expected ' + str(num_inputs) + ' inputs but received ' + str(n) +' inputs.')

    def compute_output_shape(self, inputs):
        if type(inputs) is not list:
            return _get_shape(inputs)
        if len(inputs) == 0:
            return None
        input_shapes = list(map(_get_shape, inputs))
        output_shape = input_shapes[0]
        for input_shape in input_shapes[1:]:
            output_shape = self._compute_elemwise_op_output_shape(output_shape, input_shape)
        return output_shape

    def _get_dtype(self, value):
        dtype = getattr(value, 'dtype', type(value))
        if dtype is not None and type(dtype) is not str:
            if hasattr(dtype, 'name'):
                dtype = dtype.name
            elif hasattr(dtype, '__name__'):
                dtype = dtype.__name__
            else:
                dtype = str(dtype)
        return dtype

    def compute_output_dtype(self, inputs):
        if type(inputs) is not list:
            return self._get_dtype(inputs)
        input_dtypes = [self._get_dtype(input) for input in inputs]
        if not input_dtypes:
            return None

        def _get_big_dtype(dtype1, dtype2):
            if dtype1 == dtype2:
                return dtype1
            if None in [dtype1, dtype2]:
                return None
            dtypes = np.core.numerictypes.__dict__.keys()
            dt1_ok = dtype1 in dtypes
            dt2_ok = dtype2 in dtypes
            if not dt1_ok and not dt2_ok:
                return None
            if not dt1_ok:
                return dtype2
            if not dt2_ok:
                return dtype1
            if np.dtype(dtype1) > np.dtype(dtype2):
                return dtype1
            else:
                return dtype2
        output_dtype = input_dtypes[0]
        for input_dtype in input_dtypes:
            output_dtype = _get_big_dtype(output_dtype, input_dtype)
        return output_dtype

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.

        # Arguments
            shape1: tuple or None. Shape of the first tensor
            shape2: tuple or None. Shape of the second tensor

        # Returns
            expected output shape when an element-wise operation is
            carried out on 2 tensors with shapes shape1 and shape2.
            tuple or None.

        # Raises
            ValueError: if shape1 and shape2 are not compatible for
                element-wise operations.
        """
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif len(shape2) == 0:
            return shape1
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError('Operands could not be broadcast '
                                     'together with shapes ' +
                                     str(shape1) + ' ' + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)


def get_op(func, output_shape=None, output_dtype=None, num_inputs=None, arguments={}):
    op = Op(arguments=arguments)
    if output_shape:
        op.compute_output_shape = output_shape
    if output_dtype:
        op.compute_output_dtype = output_dtype
    op.num_inputs = num_inputs
    op.call = func
    return op
