from .op import Op
py_slice = slice

class Add(Op):

    def __init__(self):
        self.num_inputs = '2+'
        super(Add, self).__init__()

    def call(self, x):
        z = x[0]
        for i in x[1:]:
            z = z + i
        return z


class Multiply(Op):

    def __init__(self):
        self.num_inputs = '2+'
        super(Multiply, self).__init__()

    def call(self, x):
        z = x[0]
        for i in x[1:]:
            z = z * i
        return z


class Subtract(Op):

    def __init__(self):
        self.num_inputs = 2
        super(Subtract, self).__init__()

    def call(self, x):
        return x[0] - x[1]


class Divide(Op):

    def __init__(self):
        self.num_inputs = 2
        super(Divide, self).__init__()

    def call(self, x):
        return x[0] / x[1]


class IntegerDivide(Op):

    def __init__(self):
        self.num_inputs = 2
        super(IntegerDivide, self).__init__()

    def call(self, x):
        return x[0] // x[1]


class Mod(Op):

    def __init__(self):
        self.num_inputs = 2
        super(Mod, self).__init__()

    def call(self, x):
        return x[0] % x[1]


class Power(Op):

    def __init__(self):
        self.num_inputs = 2
        super(Power, self).__init__()

    def call(self, x):
        return x[0] ** x[1]

class Absolute(Op):

    def __init__(self):
        self.num_inputs = 1
        super(Absolute, self).__init__()

    def call(self, x):
        return abs(x)

class Negative(Op):

    def __init__(self):
        self.num_inputs = 1
        super(Negative, self).__init__()

    def call(self, x):
        return -x

class Positive(Op):

    def __init__(self):
        self.num_inputs = 1
        super(Positive, self).__init__()

    def call(self, x):
        return +x

class LessThan(Op):

    def __init__(self):
        self.num_inputs = 2
        super(LessThan, self).__init__()

    def call(self, x):
        return x[0] < x[1]

class LessThanOrEqual(Op):

    def __init__(self):
        self.num_inputs = 2
        super(LessThanOrEqual, self).__init__()

    def call(self, x):
        return x[0] <= x[1]

class LessThan(Op):

    def __init__(self):
        self.num_inputs = 2
        super(LessThan, self).__init__()

    def call(self, x):
        return x[0] < x[1]

class LessThanOrEqual(Op):

    def __init__(self):
        self.num_inputs = 2
        super(LessThanOrEqual, self).__init__()

    def call(self, x):
        return x[0] <= x[1]

class GreaterThan(Op):

    def __init__(self):
        self.num_inputs = 2
        super(GreaterThan, self).__init__()

    def call(self, x):
        return x[0] > x[1]

class GreaterThanOrEqual(Op):

    def __init__(self):
        self.num_inputs = 2
        super(GreaterThanOrEqual, self).__init__()

    def call(self, x):
        return x[0] >= x[1]

class Equal(Op):

    def __init__(self):
        self.num_inputs = 2
        super(Equal, self).__init__()

    def call(self, x):
        return x[0] == x[1]

class Slice(Op):

    def __init__(self):
        self.num_inputs = 2
        super(Slice, self).__init__()

    def call(self, x):
        return x[0][x[1]]

    def compute_output_shape(self, inputs):
        '''Infer the shape of a tensor when it is sliced / indexed using the __getitem__ operator.
        # Arguments:
        shape: tuple. shape of the tensor
        slices: slice, int or tuple of slice and int used to slice or index the tensor.
        '''
        shape = inputs[0].shape
        slices = inputs[1]
        output_shape = []
        if type(slices) not in [list, tuple]:
            slices = [slices]
        else:
            slices = list(slices)
        while len(shape) > len(slices):
            slices += (py_slice(None),)
        for i in range(len(slices)):
            s = slices[i]
            if type(s) is list:
                output_shape.append(len(s))
            elif type(s) is py_slice:
                start = s.start
                stop = s.stop
                step = s.step
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if shape[i] is None:
                    if type(start) is int and type(stop) is int and start >= 0 and stop >= 0:
                        output_shape.append((stop - start) / step)
                    else:
                        output_shape.append(None)
                else:
                    if type(start) is int:
                        if start < 0:
                            start = shape[i] + start
                        elif start > shape[i]:
                            start = shape[i]
                    if stop is None:
                        stop = shape[i]
                    elif type(stop) is int:
                        if stop < 0:
                            stop = shape[i] + stop
                        elif stop > shape[i]:
                            stop = shape[i]
                    if set(map(type, [start, stop, step])) == set([int]):
                        output_shape.append((stop - start) / step)
                    else:
                        output_shape.append(None)
            elif type(s) is int:
                if type(shape[i]) is int and s >= shape[i]:
                    raise Exception('Index ' + str(s) + ' is out of bounds for axis ' + str(i) + ' with size '+ str(shape[i]))
                else:
                    continue
            else:
                return None
        for i in range(len(output_shape)):
            if output_shape[i] and output_shape[i] < 0:
                output_shape[i] = 0
        return tuple(output_shape)


# functional interface
all_ops = [cls for cls in globals().values() if type(cls) is type and issubclass(cls, Op)]
def _to_lower(x):
    y = ''
    for i in range(len(x)):
        c = x[i]
        if c.isupper():
            if i:
                y += '_'
        y += c.lower()
    return y
for op in all_ops:
    op_name = op.__name__
    exec('def ' + _to_lower(op_name) + '(*args):return ' + op_name + '()(args)')
del _to_lower
del all_ops
del op
del op_name


###useful abbrvts.###
lt = less_than
le = less_than_or_equal
gt = greater_than
ge = greater_than_or_equal
eq = equal
