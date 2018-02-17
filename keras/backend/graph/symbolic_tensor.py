class SymbolicTensor(object):

    @classmethod
    def __instancecheck__(cls, instance):
            return isinstance(instance, User)

    def __init__(self, function=None, inputs=None, num_outputs=None, **kwargs):
        self.function = function
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.greedy = True
        self.kwargs = kwargs

    def eval(self):
        try:
            return self.value
        except AttributeError:
            inputs = self.inputs
            if self.greedy:
                if type(inputs) is list:
                    inputs = inputs[:]
                    for i in range(len(inputs)):
                        if type(inputs[i]) is SymbolicTensor:
                            inputs[i] = inputs[i].eval()
                elif type(inputs) is SymbolicTensor:
                    inputs = inputs.eval()
            self.value = self.function(inputs, **self.kwargs)
            return self.value

    def set_value(self, value):
        self.value = value

    def __getitem__(self, *args):
        raise Exception
        def f(x, args):
            g = x.eval().__getitem__
            return g(*args)
        st = SymbolicTensor(f, self, args=args)
        st.greedy = False
        return st

    def __len__(self):
        if self.num_outputs:
            return self.num_outputs
        return 0

    def __str__(self):
        return 'SymbolicTensor'

    def __repr__(self):
        return 'SymbolicTensor'