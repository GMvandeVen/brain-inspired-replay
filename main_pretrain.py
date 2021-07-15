#!/usr/bin/env python3
import os
import numpy as np
import torch
from data.load import get_singletask_experiment
import utils
from param_stamp import get_param_stamp
from eval import evaluate
from eval import callbacks as cb
from visual import plt
import train
import options
import define_models as define




## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': True, 'only_MNIST': False, 'generative': False, 'compare_code': 'none',
              'train_options': 'all'}
    # Define input options
    parser = options.define_args(filename="main_pretrain", description='Train classifier for pretraining conv-layers.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args



## Function for running one experiment
def run(args):

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # Report whether cuda is used
    print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Create plots-directory if needed
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    print("\nPreparing the data...")
    (trainset, testset), config = get_singletask_experiment(
        name=args.experiment, data_dir=args.d_dir, verbose=True,
        normalize = True if utils.checkattr(args, "normalize") else False,
        augment = True if utils.checkattr(args, "augment") else False,
    )

    # Specify "data-loader" (among others for easy random shuffling and 'batchifying')
    train_loader = utils.get_data_loader(trainset, batch_size=args.batch, cuda=cuda, drop_last=True)

    # Determine number of iterations / epochs:
    iters = args.iters if args.iters else args.epochs*len(train_loader)
    epochs = ((args.iters-1) // len(train_loader)) + 1 if args.iters else args.epochs


    #-------------------------------------------------------------------------------------------------#

    #-----------------#
    #----- MODEL -----#
    #-----------------#

    # Specify model
    if (utils.checkattr(args, "pre_convE") or utils.checkattr(args, "pre_convD")) and \
            (hasattr(args, "depth") and args.depth>0):
        print("\nDefining the model...")
    cnn = define.define_classifier(args=args, config=config, device=device)

    # Initialize (pre-trained) parameters
    cnn = define.init_params(cnn, args)
    # - freeze weights of conv-layers?
    if utils.checkattr(args, "freeze_convE"):
        for param in cnn.convE.parameters():
            param.requires_grad = False
        cnn.convE.eval()  #--> needed to ensure batchnorm-layers also do not change
    # - freeze weights of representation-learning layers?
    if utils.checkattr(args, "freeze_full"):
        for param in cnn.parameters():
            param.requires_grad = False
        for param in cnn.classifier.parameters():
            param.requires_grad = True

    # Set optimizer
    optim_list = [{'params': filter(lambda p: p.requires_grad, cnn.parameters()), 'lr': args.lr}]
    cnn.optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Get parameter-stamp
    print("\nParameter-stamp...")
    param_stamp = get_param_stamp(args, cnn.name, verbose=True)

    # Print some model-characteristics on the screen
    utils.print_model_info(cnn, title="CLASSIFIER")

    # Define [progress_dicts] to keep track of performance during training for storing and for later plotting in pdf
    progress_dict = evaluate.initiate_progress_dict(n_tasks=1)

    # Prepare for plotting in visdom
    graph_name = cnn.name
    visdom = None if (not args.visdom) else {'env': args.experiment, 'graph': graph_name}

    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Determine after how many iterations to evaluate the model
    eval_log = args.acc_log if (args.acc_log is not None) else len(train_loader)

    # Define callback-functions to evaluate during training
    # -loss
    loss_cbs = [cb._solver_loss_cb(log=args.loss_log, visdom=visdom, epochs=epochs)]
    # -accuracy
    eval_cb = cb._eval_cb(log=eval_log, test_datasets=[testset], visdom=visdom, progress_dict=progress_dict)
    # -visualize extracted representation
    latent_space_cb = cb._latent_space_cb(log=min(5*eval_log, iters), datasets=[testset], visdom=visdom,
                                          sample_size=400)


    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- (PRE-)TRAINING -----#
    #--------------------------#

    # (Pre)train model
    print("\nTraining...")
    train.train(cnn, train_loader, iters, loss_cbs=loss_cbs, eval_cbs=[eval_cb, latent_space_cb],
                save_every=1000 if args.save else None, m_dir=args.m_dir, args=args)

    # Save (pre)trained model
    if args.save:
        # -conv-layers
        save_name = cnn.convE.name if (
            not hasattr(args, 'convE_stag') or args.convE_stag=="none"
        ) else "{}-{}".format(cnn.convE.name, args.convE_stag)
        utils.save_checkpoint(cnn.convE, args.m_dir, name=save_name)
        # -full model
        save_name = cnn.name if (
            not hasattr(args, 'full_stag') or args.full_stag=="none"
        ) else "{}-{}".format(cnn.name, args.full_stag)
        utils.save_checkpoint(cnn, args.m_dir, name=save_name)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # if requested, generate pdf.
    if args.pdf:
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = plt.open_pdf(plot_name)
        # -Fig1: show some images
        images, _ = next(iter(train_loader))            #--> get a mini-batch of random training images
        plt.plot_images_from_tensor(images, pp, title="example input images", config=config)
        # -Fig2: accuracy
        figure = plt.plot_lines(progress_dict["all_tasks"], x_axes=progress_dict["x_iteration"],
                                line_names=['ave accuracy'], xlabel="Iterations", ylabel="Test accuracy")
        pp.savefig(figure)
        # -close pdf
        pp.close()
        # -print name of generated plot on screen
        print("\nGenerated plot: {}\n".format(plot_name))



if __name__ == '__main__':
    args = handle_inputs()
    run(args)