
def updateGradInput(self, input, gradOutput):
    """
    Computing the gradient of the module with respect to its own input.
    This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

    The shape of `gradInput` is always the same as the shape of `input`.

    Make sure to both store the gradients in `gradInput` field and return it.
    """

    # The easiest case:

    # self.gradInput = gradOutput
    # return self.gradInput

    pass

def accGradParameters(self, input, gradOutput):
    """
    Computing the gradient of the module with respect to its own parameters.
    No need to override if module has no parameters (e.g. ReLU).
    """
    pass

def zeroGradParameters(self):
    """
    Zeroes `gradParams` variable if the module has params.
    """
    pass

def getParameters(self):
    """
    Returns a list with its parameters.
    If the module does not have parameters return empty list.
    """
    return []

def getGradParameters(self):
    """
    Returns a list with gradients with respect to its parameters.
    If the module does not have parameters return empty list.
    """
    return []

def train(self):
    """
    Sets training mode for the module.
    Training and testing behaviour differs for Dropout, BatchNorm.
    """
    self.training = True

def evaluate(self):
    """
    Sets evaluation mode for the module.
    Training and testing behaviour differs for Dropout, BatchNorm.
    """
    self.training = False

def __repr__(self):
    """
    Pretty printing. Should be overrided in every module if you want
    to have readable description.
    """
    return "Module"


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially.

         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`.
    """

    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})


        Just write a little loop.
        """
        self.output = input
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)


        !!!

        To ech module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.

        !!!

        """
        modules_inputs = [input] + [module.output for module in self.modules[:-1]]
        self.gradInput = gradOutput
        for module, module_input in zip(self.modules[::-1], modules_inputs[::-1]):
            self.gradInput = module.backward(module_input, self.gradInput)
        return self.gradInput


    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self ,x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        # This is a nice initialization
        stdv = 1 . /np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        # self.output = ...
        self.output = input.dot(self.W.T) + self.b
        return self.output

    def updateGradInput(self, input, gradOutput):
        # self.gradInput = ...
        self.gradInput = gradOutput.dot(self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        # self.gradW = ... ; self.gradb = ...
        self.gradW = gradOutput.T.dot(input)
        self.gradb = gradOutput.sum(axis=0)
        # pass

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1], s[0])
        return q


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        np.exp(self.output, out=self.output)
        np.divide(self.output, self.output.sum(axis=1, keepdims=True), out=self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.output
        np.subtract(self.gradInput, self.output * self.gradInput.sum(axis=1, keepdims=True), out=self.gradInput)
        return self.gradInput

    def __repr__(self):
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        np.subtract(self.output, np.log(np.exp(self.output).sum(axis=1, keepdims=True)), out=self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput - np.exp(self.output) * np.sum(gradOutput, axis=1, keepdims=True)
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateMeanVariance(self, batch_mean, batch_variance):
        self.moving_mean = batch_mean if self.moving_mean is None else self.moving_mean
        self.moving_variance = batch_variance if self.moving_variance is None else self.moving_variance

        np.multiply(self.moving_mean, self.alpha, out=self.moving_mean)
        np.multiply(batch_mean, 1 - self.alpha, out=batch_mean)
        np.add(self.moving_mean, batch_mean, out=self.moving_mean)

        np.multiply(self.moving_variance, self.alpha, out=self.moving_variance)
        np.multiply(batch_variance, 1 - self.alpha, out=batch_variance)
        np.add(self.moving_variance, batch_variance, out=self.moving_variance)

    def updateOutput(self, input):
        batch_mean = np.mean(input, axis=0) if self.training else self.moving_mean
        batch_variance = np.var(input, axis=0) if self.training else self.moving_variance
        self.output = np.divide(input - batch_mean, np.sqrt(batch_variance + self.EPS))
        if self.training:
            self.updateMeanVariance(batch_mean, batch_variance)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        batch_mean = np.mean(input, axis=0) if self.training else self.moving_mean
        batch_variance = np.var(input, axis=0) if self.training else self.moving_variance
        m = input.shape[0]

        variable0 = input - batch_mean
        variable1 = np.sum(gradOutput * variable0, axis=0)
        variable2 = np.sum(gradOutput, axis=0)
        variable3 = np.sqrt(batch_variance + self.EPS)

        self.gradInput = gradOutput / variable3
        self.gradInput -= variable1 * variable0 / m / variable3 / (batch_variance + self.EPS)
        self.gradInput -= variable2 / m / variable3
        self.gradInput += variable1 * np.sum(variable0, 0) / m ** 2 / variable3 ** (3 / 2)
        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"


class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """

    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1. / np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput * input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, input):
        if not self.training:
            self.output = input
            return self.output
        self.mask = np.random.choice(2, input.shape, p=[self.p, (1.0 - self.p)])

        np.divide(np.multiply(input, self.mask), (1 - self.p))
        self.output = np.divide(np.multiply(input, self.mask), (1 - self.p))
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.divide(np.multiply(gradOutput, self.mask), (1 - self.p))
        return self.gradInput

    def __repr__(self):
        return "Dropout"


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def updateOutput(self, input):
        routine = np.maximum if self.slope <= 1. else np.minimum
        self.output = routine(input, input * self.slope)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, gradOutput, self.slope * gradOutput)
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, input):
        self.output = np.where(input > 0, input, self.alpha * (np.exp(input) - 1.))
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, gradOutput, self.alpha * np.exp(input) * gradOutput)
        return self.gradInput

    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        self.output = np.log(np.exp(input) + 1.)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.divide(gradOutput, np.add(np.exp(-input), 1.))
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"


class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Criterion"


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15

    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()

    def updateOutput(self, input, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        self.output = -np.sum(target * np.log(input_clamp)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        self.gradInput = -target / input_clamp / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterionUnstable"


class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = -np.sum(target * input) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = -target / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"


def sgd_momentum(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('accumulated_grads', {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)

            current_var -= old_grad
            var_index += 1


def adam_optimizer(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)  # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()

    var_index = 0
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2'] ** state['t']) / (1 - config['beta1'] ** state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(config['beta1'] * var_first_moment, (1 - config['beta1']) * gradients[0][0], out=var_first_moment)
            np.add(config['beta2'] * var_second_moment, (1 - config['beta2']) * gradients[0][0] ** 2,
                   out=var_second_moment)

            current_var -= (lr_t * var_first_moment / (np.sqrt(var_second_moment) + config['epsilon']))
            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1