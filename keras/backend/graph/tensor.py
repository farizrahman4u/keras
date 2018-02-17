import inspect

tensors = {}

class Tensor(object):

    def __init__(self, shape=None, ndim=None, dtype=None, name=None):
        self.nodes = []
        global tensors
        if name is None:
            name = self.__class__.__name__
            idx = 0
            while name + '_' + str(idx) in tensors:
                idx += 1
            self.name = name + '_' + str(idx)
            tensors[self.name] = self
        else:
            if name in tensors:
                raise Exception('Another tensor with name \'' + name + '\' already exists.')
            self.name = name
            tensors[name] = self
        if shape is not None:
            self.shape = shape
        elif ndim is not None:
            self.shape = (None,) * ndim
        else:
            self.shape = None
        self.dtype = dtype

    def eval(self):
        if not hasattr(self, 'value') or self.value is None:
            if not hasattr(self, 'op') or self.op is None:
                raise Exception('Input tensor was not provided value.')
            if type(self.inputs) is list:
                inputs = [evaluate(x) for x in self.inputs]
            else:
                inputs = evaluate(self.inputs)
            args = self.op.arguments
            args_type = type(args)
            if args_type in [list, tuple]:
                y = self.op.call(inputs, *args)
            elif args_type is dict:
                y = self.op.call(inputs, **args)
            else:
                y = self.op.call(inputs)
            self.set_value(y)
        return self.value

    def set_value(self, value):
        self.value = value
        if value is None:
            self.dtype = None
            self.shape = None
        else:
            self.dtype = self._get_dtype(value)
            self.shape = self._get_shape(value)
        [node.ping(self) for node in self.nodes]

    def _get_shape(self, value):
        if type(value) is list:
            return list(map(self._get_shape, value))
        if hasattr(value, 'shape'):
            shape = value.shape
        elif hasattr(value, 'size'):
            shape = tuple(value.size())
        else:
            shape = None
        return shape

    def _get_dtype(self, value):
        if type(value) is list:
            return list(map(self._get_dtype, value))
        dtype = getattr(value, 'dtype', type(value))
        if dtype is not None and type(dtype) is not str:
            if hasattr(dtype, 'name'):
                dtype = dtype.name
            elif hasattr(dtype, '__name__'):
                dtype = dtype.__name__
            else:
                dtype = str(dtype)
        return dtype

    def __add__(self, x):
        from .magic_ops import add
        return add(self, x)

    def __radd__(self, x):
        from .magic_ops import add
        return add(x, self)

    def __mul__(self, x):
        from .magic_ops import multiply
        return multiply(self, x)

    def __rmul__(self, x):
        from .magic_ops import multiply
        return multiply(x, self)

    def __sub__(self, x):
        from .magic_ops import subtract
        return subtract(self, x)

    def __rsub__(self, x):
        from .magic_ops import subtract
        return subtract(x, self)

    def __div__(self, x):
        from .magic_ops import divide
        return divide(self, x)

    def __rdiv__(self, x):
        from .magic_ops import divide
        return divide(x, self)

    def __floordiv__(self, x):
        from .magic_ops import integer_divide
        return integer_divide(self, x)

    def __rfloordiv__(self, x):
        from .magic_ops import integer_divide
        return integer_divide(x, self)

    def __mod__(self, x):
        from .magic_ops import mod
        return mod(self, x)

    def __rmod__(self, x):
        from .magic_ops import mod
        return mod(x, self)

    def __div__(self, x):
        from .magic_ops import divide
        return divide(self, x)

    def __rdiv__(self, x):
        from .magic_ops import divide
        return divide(x, self)

    def __pow__(self, x):
        from .magic_ops import power
        return power(self, x)

    def __rpow__(self, x):
        from .magic_ops import power
        return power(x, self)

    def __neg__(self):
        from.magic_ops import negative
        return negative(self)

    def __pos__(self):
        from.magic_ops import positive
        return positive(self)

    def __abs__(self):
        from.magic_ops import absolute
        return absolute(self)

    def __lt__(self, x):
        from .magic_ops import lt
        return lt(self, x)  

    def __le__(self, x):
        from .magic_ops import le
        return le(self, x)  

    def __gt__(self, x):
        from .magic_ops import gt
        return gt(self, x)

    def __ge__(self, x):
        from .magic_ops import ge
        return ge(self, x)

    def __getitem__(self, x):
        from .magic_ops import slice
        return slice(self, x)

def evaluate(x):
    if isinstance(x, Tensor):
        return x.eval()
    else:
            x_type = type(x)
            if x_type is list:
                return list(map(evaluate, x))
            elif x_type is tuple:
                return tuple(list(map(evaluate, x))) 
            elif x_type is dict:
                return {evaluate(k):evaluate(x[k]) for k in x}
            elif x_type is slice:
                return slice(*(list(map(evaluate, [x.start, x.stop, x.step]))))
            else:
                return x
