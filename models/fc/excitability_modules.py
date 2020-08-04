import math
import torch
from torch import nn
from torch.nn.parameter import Parameter


def linearExcitability(input, weight, excitability=None, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = c(xA^T) + b`.

    Shape:
        - input:        :math:`(N, *, in\_features)`
        - weight:       :math:`(out\_features, in\_features)`
        - excitability: :math:`(out\_features)`
        - bias:         :math:`(out\_features)`
        - output:       :math:`(N, *, out\_features)`
    (NOTE: `*` means any number of additional dimensions)
    """
    if excitability is not None:
        output = input.matmul(weight.t()) * excitability
    else:
        output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output


class LinearExcitability(nn.Module):
    '''Applies a linear transformation to the incoming data: :math:`y = c(Ax) + b`

    Args:
        in_features:    size of each input sample
        out_features:   size of each output sample
        bias:           if 'False', layer will not learn an additive bias-parameter (DEFAULT=True)
        excitability:   if 'False', layer will not learn a multiplicative excitability-parameter (DEFAULT=True)

    Shape:
        - input:    :math:`(N, *, in\_features)` where `*` means any number of additional dimensions
        - output:   :math:`(N, *, out\_features)` where all but the last dimension are the same shape as the input.

    Attributes:
        weight:         the learnable weights of the module of shape (out_features x in_features)
        excitability:   the learnable multiplication terms (out_features)
        bias:           the learnable bias of the module of shape (out_features)
        excit_buffer:   fixed multiplication variable (out_features)

    Examples::

        >>> m = LinearExcitability(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    '''

    def __init__(self, in_features, out_features, bias=True, excitability=False, excit_buffer=False):
        super(LinearExcitability, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if excitability:
            self.excitability = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('excitability', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if excit_buffer:
            buffer = torch.Tensor(out_features).uniform_(1,1)
            self.register_buffer("excit_buffer", buffer)
        else:
            self.register_buffer("excit_buffer", None)
        self.reset_parameters()

    def reset_parameters(self):
        '''Modifies the parameters "in-place" to reset them at appropriate initialization values'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.excitability is not None:
            self.excitability.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        '''Running this model's forward step requires/returns:
        INPUT: -[input]: [batch_size]x[...]x[in_features]
        OUTPUT: -[output]: [batch_size]x[...]x[hidden_features]'''
        if self.excit_buffer is None:
            excitability = self.excitability
        elif self.excitability is None:
            excitability = self.excit_buffer
        else:
            excitability = self.excitability*self.excit_buffer
        return linearExcitability(input, self.weight, excitability, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'