import numpy as np
import torch
from torch.nn import functional as F
from models.utils import loss_functions as lf, modules
from models.conv.nets import ConvLayers
from models.fc.layers import fc_layer
from models.fc.nets import MLP
from models.cl.continual_learner import ContinualLearner


class Classifier(ContinualLearner):
    '''Model for encoding (i.e., feature extraction) and classifying images, enriched as "ContinualLearner"--object.'''

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, h_dim=400, fc_drop=0, fc_bn=True, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False,
                 # -training-specific settings (can be changed after setting up model)
                 hidden=False):

        # model configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop

        # settings for training
        self.hidden = hidden
        #--> if True, [self.classify] & replayed data of [self.train_a_batch] expected to be "hidden data"

        # optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ######------SPECIFY MODEL------######
        #--> convolutional layers
        self.convE = ConvLayers(
            conv_type=conv_type, block_type="basic", num_blocks=num_blocks, image_channels=image_channels,
            depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
            global_pooling=global_pooling, gated=conv_gated, output="none" if no_fnl else "normal",
        )
        self.flatten = modules.Flatten()
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        if fc_layers<2:
            self.fc_layer_sizes = [self.conv_out_units]  #--> this results in self.fcE = modules.Identity()
        elif fc_layers==2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units]+[int(x) for x in np.linspace(fc_units, h_dim, num=fc_layers-1)]
        self.units_before_classifier = h_dim if fc_layers>1 else self.conv_out_units
        #------------------------------------------------------------------------------------------#
        #--> fully connected layers
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                       excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated)#, output="none") ## NOTE: temporary change!!!
        #--> classifier
        self.classifier = fc_layer(self.units_before_classifier, classes, excit_buffer=True, nl='none', drop=fc_drop)


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.depth>0 and self.fc_layers>1:
            return "{}_{}_c{}".format(self.convE.name, self.fcE.name, self.classes)
        elif self.depth>0:
            return "{}_{}c{}".format(self.convE.name, "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                     self.classes)
        elif self.fc_layers>1:
            return "{}_c{}".format(self.fcE.name, self.classes)
        else:
            return "i{}_{}c{}".format(self.fc_layer_sizes[0], "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                      self.classes)


    def forward(self, x):
        hidden_rep = self.convE(x)
        final_features = self.fcE(self.flatten(hidden_rep))
        return self.classifier(final_features)

    def input_to_hidden(self, x):
        '''Get [hidden_rep]s (inputs to final fully-connected layers) for images [x].'''
        return self.convE(x)

    def hidden_to_output(self, hidden_rep):
        '''Map [hidden_rep]s to outputs (i.e., logits for all possible output-classes).'''
        return self.classifier(self.fcE(self.flatten(hidden_rep)))

    def feature_extractor(self, images, from_hidden=False):
        return self.fcE(self.flatten(images if from_hidden else self.convE(images)))
        #return self.classifier(self.fcE(self.flatten(images if from_hidden else self.convE(images))))

    def classify(self, x, not_hidden=False):
        '''For input [x] (image or extracted "intermediate" image features), return all predicted "scores"/"logits".'''
        image_features = self.flatten(x) if (self.hidden and not not_hidden) else self.flatten(self.convE(x))
        hE = self.fcE(image_features)
        return self.classifier(hE)


    def train_a_batch(self, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None,
                      task=1, replay_not_hidden=False, freeze_convE=False, **kwargs):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]                 <tensor> batch of corresponding labels
        [x_]                None or (<list> of) <tensor> batch of replayed inputs
                              NOTE: expected to be as [self.hidden] or [replay_up_to], unless [replay_not_hidden]==True
        [y_]                None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]           None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]               <number> in [0,1], relative importance of new task
        [active_classes]    None or (<list> of) <list> with "active" classes
        [task]              <int>, for setting task-specific mask
        [replay_not_hidden] <bool> provided [x_] are original images, even though other level might be expected'''

        # Set model to training-mode
        self.train()
        if freeze_convE:
            # - if conv-layers are frozen, they shoud be set to eval() to prevent batch-norm layers from changing
            self.convE.eval()

        # Reset optimizer
        self.optimizer.zero_grad()


        ##--(1)-- CURRENT DATA --##

        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model
            y_hat = self(x)
            # -if needed (e.g., "class" or "task" scenario), remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate multiclass prediction loss
            if y is not None and len(y.size())==0:
                y = y.expand(1)  #--> hack to make it work if batch-size is 1
            predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='none')
            # --> no reduction needed, summing over classes is "implicit"
            predL = None if y is None else lf.weighted_average(predL, weights=None, dim=0)  # -> average over batch

            # Weigh losses
            loss_cur = predL

            # Calculate training-accuracy
            accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # If XdG is combined with replay, backward-pass needs to be done before new task-mask is applied
            if (self.mask_dict is not None) and (x_ is not None):
                weighted_current_loss = rnt*loss_cur
                weighted_current_loss.backward()
        else:
            accuracy = predL = None
            # -> it's possible there is only "replay" [i.e., for offline with incremental task learning scenario]


        ##--(2)-- REPLAYED DATA --##

        if x_ is not None:
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            TaskIL = (type(y_)==list) if (y_ is not None) else (type(scores_)==list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [torch.tensor(0., device=self._device())]*n_replays
            predL_r = [torch.tensor(0., device=self._device())]*n_replays
            distilL_r = [torch.tensor(0., device=self._device())]*n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            if (not type(x_)==list) and (self.mask_dict is None):
                y_hat_all = self.classify(x_, not_hidden=replay_not_hidden)

            # Loop to perform each replay
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_) == list) or (self.mask_dict is not None):
                    x_temp_ = x_[replay_id] if type(x_) == list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id + 1)
                    y_hat_all = self.classify(x_temp_, not_hidden=replay_not_hidden)

                # -if needed (e.g., "class" or "task" scenario), remove predictions for classes not in replayed task
                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='none')
                    predL_r[replay_id] = lf.weighted_average(predL_r[replay_id], dim=0)
                    #-> average over batch
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                    n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
                    distilL_r[replay_id] = lf.loss_fn_kd(
                        scores=y_hat[:, :n_classes_to_consider], target_scores=scores_[replay_id], T=self.KD_temp,
                    )  # --> summing over classes & averaging over batch within this function
                # Weigh losses
                if self.replay_targets=="hard":
                    loss_replay[replay_id] = predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] = distilL_r[replay_id]

                # If task-specific mask, backward pass needs to be performed before next task-mask is applied
                if self.mask_dict is not None:
                    weighted_replay_loss_this_task = (1-rnt) * loss_replay[replay_id] / n_replays
                    weighted_replay_loss_this_task.backward()

        # Calculate total loss
        loss_replay = None if (x_ is None) else sum(loss_replay)/n_replays
        loss_total = loss_replay if (x is None) else (loss_cur if x_ is None else rnt*loss_cur+(1-rnt)*loss_replay)


        ##--(3)-- ALLOCATION LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c>0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda>0:
            loss_total += self.ewc_lambda * ewc_loss


        # Backpropagate errors (if not yet done)
        if (self.mask_dict is None) or (x_ is None):
            loss_total.backward()
        # Take optimization-step
        self.optimizer.step()


        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item()/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item()/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'ewc': ewc_loss.item(), 'si_loss': surrogate_loss.item(),
            'accuracy': accuracy if accuracy is not None else 0.,
        }

