import numpy as np
from sklearn import manifold
import torch
import visual.visdom
import visual.plt
import utils


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####

def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             no_task_mask=False, task=None):
    '''Evaluate the accuracy (= proportion of samples classified correctly) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Set model to eval()-mode
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(device), labels.to(device)
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        with torch.no_grad():
            scores = model.classify(data, not_hidden=True)
            scores = scores if (allowed_classes is None) else scores[:, allowed_classes]
            _, predicted = torch.max(scores, 1)
        # -update statistics
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
    accuracy = total_correct / total_tested

    # Print result on screen (if requested) and return it
    if verbose:
        print('=> accuracy: {:.3f}'.format(accuracy))
    return accuracy


def initiate_progress_dict(n_tasks):
    '''Initiate <dict> with all accuracy-measures to keep track of.'''
    progress_dict = {}
    progress_dict["all_tasks"] = [[] for _ in range(n_tasks)]
    progress_dict["average"] = []
    progress_dict["x_iteration"] = []
    progress_dict["x_task"] = []
    return progress_dict


def test_accuracy(model, datasets, current_task, iteration, classes_per_task=None, scenario="none",
                  progress_dict=None, test_size=None, visdom=None, verbose=False, no_task_mask=False):
    '''Evaluate accuracy of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [progress_dict]    None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    [scenario]          <str> for how to decide which classes to include during evaluation
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    n_tasks = len(datasets)
    accs = []
    for i in range(n_tasks):
        if i+1 <= current_task:
            if scenario=='task':
                allowed_classes = list(range(classes_per_task*i, classes_per_task*(i+1)))
            elif scenario=='class':
                allowed_classes = list(range(classes_per_task*(current_task)))
            else:
                allowed_classes = None
            accs.append(validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                  allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i+1))
        else:
            accs.append(0)
    average_accs = sum(
        [accs[task_id] if task_id==0 else accs[task_id] for task_id in range(current_task)]
    ) / (current_task)

    # Print results on screen
    if verbose:
        print(' => ave accuracy: {:.3f}'.format(average_accs))

    # Send results to visdom server
    names = ['task {}'.format(i + 1) for i in range(n_tasks)]
    if visdom is not None:
        visual.visdom.visualize_scalars(
            scalars=accs, names=names, iteration=iteration,
            title="Accuracy per task ({})".format(visdom["graph"]), env=visdom["env"], ylabel="accuracy"
        )
        if n_tasks>1:
            visual.visdom.visualize_scalars(
                scalars=[average_accs], names=["ave accuracy"], iteration=iteration,
                title="Average accuracy ({})".format(visdom["graph"]), env=visdom["env"], ylabel="accuracy"
            )

    # Append results to [progress]-dictionary and return
    if progress_dict is not None:
        for task_id, _ in enumerate(names):
            progress_dict["all_tasks"][task_id].append(accs[task_id])
        progress_dict["average"].append(average_accs)
        progress_dict["x_iteration"].append(iteration)
        progress_dict["x_task"].append(current_task)
    return progress_dict



####--------------------------------------------------------------------------------------------------------------####

####------------------------------------------####
####----VISUALIZE EXTRACTED REPRESENTATION----####
####------------------------------------------####

def visualize_latent_space(model, X, y=None, visdom=None, pdf=None, verbose=False):
    '''Show T-sne projection of feature representation used to classify from (with each class in different color).'''

    # Set model to eval()-mode
    model.eval()

    # Compute the representation used for classification
    if verbose:
        print("Computing feature space...")
    with torch.no_grad():
        z_mean = model.feature_extractor(X)

    # Compute t-SNE embedding of latent space (unless z has 2 dimensions!)
    if z_mean.size()[1]==2:
        z_tsne = z_mean.cpu().numpy()
    else:
        if verbose:
            print("Computing t-SNE embedding...")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        z_tsne = tsne.fit_transform(z_mean.cpu())

    # Plot images according to t-sne embedding
    if pdf is not None:
        figure = visual.plt.plot_scatter(z_tsne[:, 0], z_tsne[:, 1], colors=y)
        pdf.savefig(figure)
    if visdom is not None:
        message = ("Visualization of extracted representation")
        visual.visdom.scatter_plot(z_tsne, title="{} ({})".format(message, visdom["graph"]),
                                   colors=y+1 if y is not None else y, env=visdom["env"])



####--------------------------------------------------------------------------------------------------------------####

####----------------------------####
####----GENERATOR EVALUATION----####
####----------------------------####

def show_samples(model, config, pdf=None, visdom=None, size=32, sample_mode=None, title="Generated samples",
                 allowed_classes=None, allowed_domains=None):
    '''Plot samples from a generative model in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    model.eval()

    # Generate samples from the model
    sample = model.sample(size, sample_mode=sample_mode, allowed_classes=allowed_classes,
                          allowed_domains=allowed_domains, only_x=True)
    # -correctly arrange pixel-values and move to cpu (if needed)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)

    # Plot generated images in [pdf] and/or [visdom]
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size)))
    # -make plots
    if pdf is not None:
        visual.plt.plot_images_from_tensor(image_tensor, pdf, title=title, nrow=nrow)
    if visdom is not None:
        mode = "" if sample_mode is None else "(mode = {})".format(sample_mode)
        visual.visdom.visualize_images(
            tensor=image_tensor, env=visdom["env"], nrow=nrow,
            title='Generated samples {} ({})'.format(mode, visdom["graph"]),
        )



####--------------------------------------------------------------------------------------------------------------####

####--------------------------------####
####----RECONSTRUCTOR EVALUATION----####
####--------------------------------####

def show_reconstruction(model, dataset, config, pdf=None, visdom=None, size=32, epoch=None, task=None,
                        no_task_mask=False):
    '''Plot reconstructed examples by an auto-encoder [model] on [dataset], either in [pdf] and/or in [visdom].'''

    # Get device-type / using cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Set model to evaluation-mode
    model.eval()

    # Get data
    data_loader = utils.get_data_loader(dataset, size, cuda=cuda)
    (data, labels) = next(iter(data_loader))

    # If needed, apply correct specific task-mask (for fully-connected hidden layers in encoder)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Evaluate model
    data, labels = data.to(device), labels.to(device)
    with torch.no_grad():
        gate_input = (
            torch.tensor(np.repeat(task-1, size)).to(device) if model.dg_type=="task" else labels
        ) if (utils.checkattr(model, 'dg_gates') and model.dg_prop>0) else None
        recon_output = model(data, gate_input=gate_input, full=True, reparameterize=False)
    recon_batch = recon_output[0]

    # Plot original and reconstructed images
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size*2)))
    # -collect and arrange pixel-values
    comparison = torch.cat(
        [data.view(-1, config['channels'], config['size'], config['size'])[:size],
         recon_batch.view(-1, config['channels'], config['size'], config['size'])[:size]]
    ).cpu()
    image_tensor = comparison.view(-1, config['channels'], config['size'], config['size'])
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)
    # -make plots
    if pdf is not None:
        epoch_stm = "" if epoch is None else " after epoch ".format(epoch)
        task_stm = "" if task is None else " (task {})".format(task)
        visual.plt.plot_images_from_tensor(
            image_tensor, pdf, nrow=nrow, title="Reconstructions" + task_stm + epoch_stm
        )
    if visdom is not None:
        visual.visdom.visualize_images(
            tensor=image_tensor, title='Reconstructions ({})'.format(visdom["graph"]), env=visdom["env"], nrow=nrow,
        )