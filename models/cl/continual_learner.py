import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import get_data_loader


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract  module to add continual learning capabilities to a classifier.

    Adds methods/attributes for "context-dependent gating" (XdG), "elastic weight consolidation" (EWC),
    "synaptic intelligence" (SI) and "generative replay" (GR) to its subclasses.'''

    def __init__(self):
        super().__init__()

        # XdG:
        self.mask_dict = None        # -> <dict> with task-specific masks for each hidden fully-connected layer
        self.excit_buffer_list = []  # -> <list> with excit-buffers for all hidden fully-connected layers

        # SI:
        self.si_c = 0           #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1      #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        # EWC:
        self.ewc_lambda = 0     #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.         #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = True      #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = None    #-> number minibatches to use for estimating FI-matrix (if "None", one pass over data)
        self.EWC_task_count = 0 #-> keeps track of number of quadratic loss terms (for "offline EWC")

        # Replay:
        self.replay_targets = "hard"  # should distillation loss be used? (hard|soft)
        self.KD_temp = 2.             # temperature for distillation loss


    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass


    #----------------- XdG-specifc functions -----------------#

    def define_XdGmask(self, gating_prop, n_tasks):
        '''Define task-specific masks, by randomly selecting [gating_prop]% of nodes per hidden fc-layer for each task.

        [gating_prop]   <num>, between 0 and 1, proportion of nodes to be gated
        [n_tasks]       <int>, total number of tasks'''

        mask_dict = {}
        excit_buffer_list = []
        for task_id in range(n_tasks):
            mask_dict[task_id + 1] = {}
            for i in range(self.fcE.layers):
                layer = getattr(self.fcE, "fcLayer{}".format(i + 1)).linear
                if task_id == 0:
                    excit_buffer_list.append(layer.excit_buffer)
                n_units = len(layer.excit_buffer)
                gated_units = np.random.choice(n_units, size=int(gating_prop * n_units), replace=False)
                mask_dict[task_id + 1][i] = gated_units
        self.mask_dict = mask_dict
        self.excit_buffer_list = excit_buffer_list

    def apply_XdGmask(self, task):
        '''Apply task-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [task]   <int>, starting from 1'''

        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        # Loop over all buffers for which a task-specific mask has been specified
        for i,excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1., len(excit_buffer))
            gating_mask[self.mask_dict[task][i]] = 0.      # -> find task-specifc mask
            excit_buffer.set_(torchType.new(gating_mask))  # -> apply this mask

    def reset_XdGmask(self):
        '''Remove task-specific mask, by setting all "excit-buffers" to 1.'''
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1., len(excit_buffer))  # -> define "unit mask" (i.e., no masking at all)
            excit_buffer.set_(torchType.new(gating_mask))   # -> apply this unit mask


    #----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, dataset, allowed_classes=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index,(x,y) in enumerate(data_loader):
            # Break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # Run forward pass of model
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            # Use a weighted combination of all labels
            with torch.no_grad():
                label_weights = F.softmax(output, dim=1)  # --> get weights, which shouldn't have gradient tracked
            for label_index in range(output.shape[1]):
                label = torch.LongTensor([label_index]).to(self._device())
                negloglikelihood = F.cross_entropy(output, label)  #--> get neg log-likelihoods for this classs
                # Calculate gradient of negative loglikelihood
                self.zero_grad()
                negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
                # Square gradients and keep running sum (using the weights)
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)


    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count+1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma*fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there is no stored fisher yet
            return torch.tensor(0., device=self._device())


    #------------- "Intelligent Synapses"-specifc functions -------------#

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n]/(p_change**2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)


    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())
