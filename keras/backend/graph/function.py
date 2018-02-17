from .tensor import Tensor
from .variable import Variable


class Function(object):

	def __init__(self, inputs, outputs):
		if type(inputs) not in [list, tuple]:
			inputs = [inputs]
		if type(outputs) not in [list, tuple]:
			outputs = [outputs]
		self.inputs = inputs
		self.outputs = outputs
		self._check_disconnected_graph()

	def _check_tensor(self, tensor):
		if isinstance(tensor, Variable):
			return True
		if not isinstance(tensor, Tensor):
			return True
		if not hasattr(tensor, 'op') or tensor.op is None:
			if tensor not in self.inputs:
				return tensor
		else:
			tensor_inputs = tensor.inputs
			if type(tensor_inputs) is list:
				vals = map(self._check_tensor, tensor_inputs)
				for val in vals:
					if val != True:
						return val
				return True
			else:
				return self._check_tensor(tensor_inputs)
		return True

	def _check_disconnected_graph(self):
		ok_outputs = []
		not_ok_outputs = []
		for output in self.outputs:
			res = self._check_tensor(output)
			if res == True:
				ok_outputs.append(res)
			else:
				not_ok_outputs.append((output, res))
		if not_ok_outputs:
			err_msg = 'Disconnected graph. Unable to evaluate the following outputs :'
			for o in not_ok_outputs:
				err_msg += '\n' + str(o[0]) + ' (' + str(o[1]) + ' has no inputs in this graph.)'
			if len(ok_outputs) > 0:
				err_msg += '\n Following outputs were evaluated without issue :\n'
				err_msg += str(ok_outputs)
			raise Exception(err_msg)

	def __call__(self, x):
		if type(x) not in [list, tuple]:
			x = [x]
		len_x = len(x)
		num_inputs = len(self.inputs)
		assert len_x == num_inputs, 'Function expected ' + str(num_inputs) + ' inputs, but received ' + str(len_x) +'.'
		for i, j in zip(self.inputs, x):
			i.set_value(j)
		return [output.eval() for output in self.outputs]
