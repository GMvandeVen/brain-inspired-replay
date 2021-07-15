import visual.visdom
from . import evaluate
from torch.utils.data import ConcatDataset
import utils


#########################################################
## Callback-functions for evaluating model-performance ##
#########################################################

def _sample_cb(log, config, visdom=None, pdf=None, test_datasets=None, sample_size=64, iters_per_task=None):
    '''Initiates function for evaluating samples of generative model.

    [test_datasets]     None or <list> of <Datasets> (if provided, also reconstructions are shown)'''

    def sample_cb(generator, batch, task=1, epoch=None, allowed_classes=None, **kwargs):
        '''Callback-function, to evaluate sample (and reconstruction) ability of the model.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        if iteration % log == 0:
            statement = " after {} iters{}{}".format(batch, "" if epoch is None else " (epoch {})".format(epoch),
                                                     "" if iters_per_task is None else " in task {}".format(task))

            # Evaluate reconstruction-ability of model on [test_dataset]
            if test_datasets is not None:
                # Reconstruct samples from current task
                evaluate.show_reconstruction(generator, test_datasets[task-1], config, size=int(sample_size/2),
                                             pdf=pdf, visdom=visdom, epoch=epoch, task=task)

            # Generate samples
            evaluate.show_samples(generator, config, pdf=pdf, visdom=visdom, size=sample_size,
                                  allowed_classes=allowed_classes, allowed_domains=list(range(task)),
                                  title="Generated images{}".format(statement))

    # Return the callback-function (except if neither visdom or pdf is selected!)
    return sample_cb if ((visdom is not None) or (pdf is not None)) else None



def _eval_cb(log, test_datasets, visdom=None, progress_dict=None, iters_per_task=None, test_size=None,
             classes_per_task=None, scenario="none"):
    '''Initiates function for evaluating performance of classifier (in terms of accuracy).

    [test_datasets]     <list> of <Datasets>; also if only 1 task, it should be presented as a list!
    [classes_per_task]  <int> number of "active" classes per task
    [scenario]          <str> for how to decide which classes to include during evaluation'''

    def eval_cb(classifier, batch, task=1, **kwargs):
        '''Callback-function, to evaluate performance of classifier.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        # evaluate the solver on multiple tasks (and log to visdom)
        if iteration % log == 0:
            evaluate.test_accuracy(classifier, test_datasets, task, iteration,
                                   classes_per_task=classes_per_task, scenario=scenario, progress_dict=progress_dict,
                                   test_size=test_size, visdom=visdom)

    ## Return the callback-function (except if neither visdom or [progress_dict] is selected!)
    return eval_cb if ((visdom is not None) or (progress_dict is not None)) else None



##------------------------------------------------------------------------------------------------------------------##

##############################################################
## Callback-functions for visualing statistics of the model ##
##############################################################

def _latent_space_cb(log, datasets, visdom=None, pdf=None, sample_size=400, iters_per_task=None):
    '''Initiates function for visualizing final layer features of a classifier or VAE.

    [log]          <int>, indicating after how many iterations the callback-function should be evaluated
    [datasets]     <list> of <Datasets>'''

    def latent_space_cb(model, batch, task=1, **kwargs):
        '''Callback-function, to visualize latent space of the model.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        if iteration % log == 0:
            dataset = ConcatDataset(datasets[:task])
            loader = utils.get_data_loader(dataset, batch_size=sample_size, cuda=model._is_on_cuda(), drop_last=True)
            X,y = next(iter(loader))
            evaluate.visualize_latent_space(model, X.to(model._device()), y=y.to(model._device()), visdom=visdom,
                                            pdf=pdf)

    # Return the callback-function (except if neither visdom or pdf is selected!)
    return latent_space_cb if ((visdom is not None) or (pdf is not None)) else None



##------------------------------------------------------------------------------------------------------------------##

###############################################################
## Callback-functions for keeping track of training-progress ##
###############################################################

def _solver_loss_cb(log, visdom, model=None, tasks=None, iters_per_task=None, epochs=None, rnt=None, replay=False,
                    progress_bar=True):
    '''Initiates function for keeping track of, and reporting on, the progress of the solver's training.'''

    def cb(bar, iter, loss_dict, task=1, epoch=None):
        '''Callback-function, to call on every iteration to keep track of training progress.'''

        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        ##--------------------------------PROGRESS BAR---------------------------------##
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else " Task: {}/{} |".format(task, tasks)
            epoch_stm = "" if ((epochs is None) or (epoch is None)) else " Epoch: {}/{} |".format(epoch, epochs)
            bar.set_description(
                ' <MAIN MODEL> |{t_stm}{e_stm} training loss: {loss:.3} | training accuracy: {acc:.3} |'
                    .format(t_stm=task_stm, e_stm=epoch_stm, loss=loss_dict['loss_total'], acc=loss_dict['accuracy'])
            )
            bar.update(1)
        ##-----------------------------------------------------------------------------##

        # log the loss of the solver (to visdom)
        if (iteration % log == 0) and (visdom is not None):
            if tasks is None or tasks==1:
                # -overview of loss -- single task
                plot_data = [loss_dict['pred']]
                names = ['prediction']
            else:
                # -overview of losses -- multiple tasks
                current_rnt = (1./task if (rnt is None) or (task==1) else rnt) if replay else 1.
                plot_data = [current_rnt*loss_dict['pred']]
                names = ['pred']
                if replay:
                    if model.replay_targets=="hard":
                        plot_data += [(1-current_rnt)*loss_dict['pred_r']]
                        names += ['pred - r']
                    elif model.replay_targets=="soft":
                        plot_data += [(1-current_rnt)*loss_dict['distil_r']]
                        names += ['KD - r']
                if model.ewc_lambda>0:
                    plot_data += [model.ewc_lambda * loss_dict['ewc']]
                    names += ['EWC (lambda={})'.format(model.ewc_lambda)]
                if model.si_c>0:
                    plot_data += [model.si_c * loss_dict['si_loss']]
                    names += ['SI (c={})'.format(model.si_c)]
            visual.visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="CLASSIFIER: loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function.
    return cb



def _VAE_loss_cb(log, visdom, model, tasks=None, iters_per_task=None, epochs=None, rnt=None, replay=False,
                 progress_bar=True):
    '''Initiates functions for keeping track of, and reporting on, the progress of the generator's training.'''

    def cb(bar, iter, loss_dict, task=1, epoch=None):
        '''Callback-function, to perform on every iteration to keep track of training progress.'''

        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        ##--------------------------------PROGRESS BAR---------------------------------##
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else " Task: {}/{} |".format(task, tasks)
            epoch_stm = "" if ((epochs is None) or (epoch is None)) else " Epoch: {}/{} |".format(epoch, epochs)
            bar.set_description(
                ' <GENERATOR>  |{t_stm}{e_stm} training loss: {loss:.3} | training accuracy: {acc:.3} |'
                    .format(t_stm=task_stm, e_stm=epoch_stm, loss=loss_dict['loss_total'], acc=loss_dict['accuracy'])
            )
            bar.update(1)
        ##-----------------------------------------------------------------------------##

        # plot training loss every [log]
        if (iteration % log == 0) and (visdom is not None):
            # -overview of losses
            plot_data = list()
            names = list()
            current_rnt = (1./task if (rnt is None or task==1) else rnt) if replay else 1.
            if model.lamda_rcl>0:
                plot_data += [current_rnt*model.lamda_rcl*loss_dict['recon']]
                names += ['Recon (x{})'.format(model.lamda_rcl)]
            if model.lamda_vl>0:
                plot_data += [current_rnt*model.lamda_vl*loss_dict['variat']]
                names += ['Variat (x{})'.format(model.lamda_vl)]
            if hasattr(model, 'lamda_pl') and model.lamda_pl>0 and hasattr(model, 'classifier'):
                plot_data += [current_rnt*model.lamda_pl*loss_dict['pred']]
                names += ['Prediction (x{})'.format(model.lamda_pl)]
            if tasks is not None:
                if tasks>1:
                    if hasattr(model, 'ewc_lambda') and model.ewc_lambda>0:
                        plot_data += [model.ewc_lambda * loss_dict['ewc']]
                        names += ['EWC (lambda={})'.format(model.ewc_lambda)]
                    if hasattr(model, 'si_c') and model.si_c>0:
                        plot_data += [model.si_c * loss_dict['si_loss']]
                        names += ['SI (c={})'.format(model.si_c)]
            if tasks is not None and replay:
                if tasks>1:
                    if model.lamda_rcl > 0:
                        plot_data += [(1-current_rnt)*model.lamda_rcl * loss_dict['recon_r']]
                        names += ['Recon - r (x{})'.format(model.lamda_rcl)]
                    if model.lamda_vl > 0:
                        plot_data += [(1-current_rnt)*model.lamda_vl * loss_dict['variat_r']]
                        names += ['Variat - r (x{})'.format(model.lamda_vl)]
                    if hasattr(model, 'lamda_pl') and model.lamda_pl > 0:
                        # note that this one is weighted according to current task relative weight!
                        if model.replay_targets=="hard":
                            plot_data += [(1-current_rnt)*model.lamda_pl * loss_dict['pred_r']]
                            names += ['Prediction - r (x{})'.format(model.lamda_pl)]
                        elif model.replay_targets=="soft":
                            plot_data += [(1-current_rnt)*model.lamda_pl * loss_dict['distil_r']]
                            names += ['Distill - r (x{})'.format(model.lamda_pl)]
            visual.visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="VAE: loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function
    return cb