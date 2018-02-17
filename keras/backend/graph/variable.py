from .tensor import Tensor

def _is_num(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


_FLOATX = 'float32'

def floatx():
	return _FLOATX

class Variable(Tensor):

	def __init__(self, value, **kwargs):
		super(Variable, self).__init__(**kwargs)
		self.set_value(value)

	def eval(self):
		return self.value
