#!/usr/bin/env python3
import numpy as np
import os
from scipy.stats import entropy
import torch
from torch import optim
from torch.utils.data import ConcatDataset
from torch.nn import functional as F

# -custom-written libraries
import options
import utils
import define_models as define
from data.load import get_multitask_experiment
from eval import evaluate
from eval import callbacks as cb
import eval.precision_recall as pr
import eval.fid as fid
from train import train_cl
from param_stamp import get_param_stamp
from models.cl.continual_learner import ContinualLearner


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True, 'compare_code': 'none'}
    # Define input options
    parser = options.define_args(filename="main_cl", description='Compare & combine continual learning approaches.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args


## Function for running one continual learning experiment
def run(args, verbose=False):

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it and exit
    if args.get_stamp:
        from param_stamp import get_param_stamp_from_args
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Report whether cuda is used
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)


    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\nPreparing the data...")
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir,
        normalize=True if utils.checkattr(args, "normalize") else False,
        augment=True if utils.checkattr(args, "augment") else False,
        verbose=verbose, exception=True if args.seed<10 else False, only_test=(not args.train)
    )


    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- MAIN MODEL -----#
    #----------------------#

    # Define main model (i.e., classifier, if requested with feedback connections)
    if verbose and (utils.checkattr(args, "pre_convE") or utils.checkattr(args, "pre_convD")) and \
            (hasattr(args, "depth") and args.depth>0):
        print("\nDefining the model...")
    if utils.checkattr(args, 'feedback'):
        model = define.define_autoencoder(args=args, config=config, device=device)
    else:
        model = define.define_classifier(args=args, config=config, device=device)

    # Initialize / use pre-trained / freeze model-parameters
    # - initialize (pre-trained) parameters
    model = define.init_params(model, args)
    # - freeze weights of conv-layers?
    if utils.checkattr(args, "freeze_convE"):
        for param in model.convE.parameters():
            param.requires_grad = False
    if utils.checkattr(args, 'feedback') and utils.checkattr(args, "freeze_convD"):
        for param in model.convD.parameters():
            param.requires_grad = False

    # Define optimizer (only optimize parameters that "requires_grad")
    model.optim_list = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
    ]
    model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))


    #-------------------------------------------------------------------------------------------------#

    #----------------------------------------------------#
    #----- CL-STRATEGY: REGULARIZATION / ALLOCATION -----#
    #----------------------------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'ewc'):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        model.fisher_n = args.fisher_n
        model.online = utils.checkattr(args, 'online')
        if model.online:
            model.gamma = args.gamma

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'si'):
        model.si_c = args.si_c if args.si else 0
        model.epsilon = args.epsilon

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'xdg') and args.xdg_prop>0:
        model.define_XdGmask(gating_prop=args.xdg_prop, n_tasks=args.tasks)


    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay') and not args.replay=="none":
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = (hasattr(args, 'replay') and args.replay=="generative" and not utils.checkattr(args, 'feedback'))
    if train_gen:
        # Specify architecture
        generator = define.define_autoencoder(args, config, device, generator=True)

        # Initialize parameters
        generator = define.init_params(generator, args)
        # -freeze weights of conv-layers?
        if utils.checkattr(args, "freeze_convE"):
            for param in generator.convE.parameters():
                param.requires_grad = False
        if utils.checkattr(args, "freeze_convD"):
            for param in generator.convD.parameters():
                param.requires_grad = False

        # Set optimizer(s)
        generator.optim_list = [
            {'params': filter(lambda p: p.requires_grad, generator.parameters()),
             'lr': args.lr_gen if hasattr(args, 'lr_gen') else args.lr},
        ]
        generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
    else:
        generator = None


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp(
        args, model.name, verbose=verbose,
        replay=True if (hasattr(args, 'replay') and not args.replay=="none") else False,
        replay_model_name=generator.name if (
            hasattr(args, 'replay') and args.replay in ("generative") and not utils.checkattr(args, 'feedback')
        ) else None,
    )

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        utils.print_model_info(model, title="MAIN MODEL")
        # -generator
        if generator is not None:
            utils.print_model_info(generator, title="GENERATOR")

    # Define [progress_dict] to keep track of performance during training for storing and for later plotting in pdf
    progress_dict = evaluate.initiate_progress_dict(args.tasks)

    # Prepare for plotting in visdom
    visdom = None
    if args.visdom:
        env_name = "{exp}{tasks}-{scenario}".format(exp=args.experiment, tasks=args.tasks, scenario=args.scenario)
        replay_statement = "{mode}{fb}{con}{gat}{int}{dis}{b}{u}".format(
            mode=args.replay,
            fb="Rtf" if utils.checkattr(args, "feedback") else "",
            con="Con" if (hasattr(args, "prior") and args.prior=="GMM" and utils.checkattr(args, "per_class")) else "",
            gat="Gat{}".format(args.dg_prop) if (
                utils.checkattr(args, "dg_gates") and hasattr(args, "dg_prop") and args.dg_prop>0
            ) else "",
            int="Int" if utils.checkattr(args, "hidden") else "",
            dis="Dis" if args.replay=="generative" and args.distill else "",
            b="" if (args.batch_replay is None or args.batch_replay==args.batch) else "-br{}".format(args.batch_replay),
            u="" if args.g_fc_uni==args.fc_units else "-gu{}".format(args.g_fc_uni)
        ) if (hasattr(args, "replay") and not args.replay=="none") else "NR"
        graph_name = "{replay}{syn}{ewc}{xdg}".format(
            replay=replay_statement,
            syn="-si{}".format(args.si_c) if utils.checkattr(args, 'si') else "",
            ewc="-ewc{}{}".format(
                args.ewc_lambda,"-O{}".format(args.gamma) if utils.checkattr(args, "online") else ""
            ) if utils.checkattr(args, 'ewc') else "",
            xdg="" if (not utils.checkattr(args, 'xdg')) or args.xdg_prop==0 else "-XdG{}".format(args.xdg_prop),
        )
        visdom = {'env': env_name, 'graph': graph_name}


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    g_iters = args.g_iters if hasattr(args, 'g_iters') else args.iters

    # Callbacks for reporting on and visualizing loss
    generator_loss_cbs = [
        cb._VAE_loss_cb(log=args.loss_log, visdom=visdom, replay=(hasattr(args, "replay") and not args.replay=="none"),
                        model=model if utils.checkattr(args, 'feedback') else generator, tasks=args.tasks,
                        iters_per_task=args.iters if utils.checkattr(args, 'feedback') else g_iters)
    ] if (train_gen or utils.checkattr(args, 'feedback')) else [None]
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, iters_per_task=args.iters, tasks=args.tasks,
                           replay=(hasattr(args, "replay") and not args.replay=="none"))
    ] if (not utils.checkattr(args, 'feedback')) else [None]

    # Callbacks for evaluating and plotting generated / reconstructed samples
    no_samples = (utils.checkattr(args, "no_samples") or (
        utils.checkattr(args, "hidden") and hasattr(args, 'depth') and args.depth>0
    ))
    sample_cbs = [
        cb._sample_cb(log=args.sample_log, visdom=visdom, config=config, test_datasets=test_datasets,
                      sample_size=args.sample_n, iters_per_task=g_iters)
    ] if ((train_gen or utils.checkattr(args, 'feedback')) and not no_samples) else [None]

    # Callbacks for reporting and visualizing accuracy, and visualizing representation extracted by main model
    # -visdom (i.e., after each [acc_log]
    eval_cb = cb._eval_cb(
        log=args.acc_log, test_datasets=test_datasets, visdom=visdom, progress_dict=None, iters_per_task=args.iters,
        test_size=args.acc_n, classes_per_task=classes_per_task, scenario=args.scenario,
    )
    # -pdf / reporting: summary plots (i.e, only after each task)
    eval_cb_full = cb._eval_cb(
        log=args.iters, test_datasets=test_datasets, progress_dict=progress_dict,
        iters_per_task=args.iters, classes_per_task=classes_per_task, scenario=args.scenario,
    )
    # -visualize feature space
    latent_space_cb = cb._latent_space_cb(
        log=args.iters, datasets=test_datasets, visdom=visdom, iters_per_task=args.iters, sample_size=400,
    )
    # -collect them in <lists>
    eval_cbs = [eval_cb, eval_cb_full, latent_space_cb]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if args.train:
        if verbose:
            print("\nTraining...")
        # Train model
        train_cl(
            model, train_datasets, replay_mode=args.replay if hasattr(args, 'replay') else "none",
            scenario=args.scenario, classes_per_task=classes_per_task, iters=args.iters,
            batch_size=args.batch, batch_size_replay=args.batch_replay if hasattr(args, 'batch_replay') else None,
            generator=generator, gen_iters=g_iters, gen_loss_cbs=generator_loss_cbs,
            feedback=utils.checkattr(args, 'feedback'), sample_cbs=sample_cbs, eval_cbs=eval_cbs,
            loss_cbs=generator_loss_cbs if utils.checkattr(args, 'feedback') else solver_loss_cbs,
            args=args, reinit=utils.checkattr(args, 'reinit'), only_last=utils.checkattr(args, 'only_last')
        )
        # Save evaluation metrics measured throughout training
        file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
        utils.save_object(progress_dict, file_name)
        # Save trained model(s), if requested
        if args.save:
            save_name = "mM-{}".format(param_stamp) if (
                not hasattr(args, 'full_stag') or args.full_stag == "none"
            ) else "{}-{}".format(model.name, args.full_stag)
            utils.save_checkpoint(model, args.m_dir, name=save_name, verbose=verbose)
            if generator is not None:
                save_name = "gM-{}".format(param_stamp) if (
                    not hasattr(args, 'full_stag') or args.full_stag == "none"
                ) else "{}-{}".format(generator.name, args.full_stag)
                utils.save_checkpoint(generator, args.m_dir, name=save_name, verbose=verbose)

    else:
        # Load previously trained model(s) (if goal is to only evaluate previously trained model)
        if verbose:
            print("\nLoading parameters of the previously trained models...")
        load_name = "mM-{}".format(param_stamp) if (
            not hasattr(args, 'full_ltag') or args.full_ltag == "none"
        ) else "{}-{}".format(model.name, args.full_ltag)
        utils.load_checkpoint(model, args.m_dir, name=load_name, verbose=verbose,
                              add_si_buffers=(isinstance(model, ContinualLearner) and utils.checkattr(args, 'si')))
        if generator is not None:
            load_name = "gM-{}".format(param_stamp) if (
                not hasattr(args, 'full_ltag') or args.full_ltag == "none"
            ) else "{}-{}".format(generator.name, args.full_ltag)
            utils.load_checkpoint(generator, args.m_dir, name=load_name, verbose=verbose)


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------#
    #----- EVALUATION of CLASSIFIER-----#
    #-----------------------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate accuracy of final model on full test-set
    accs = [evaluate.validate(
        model, test_datasets[i], verbose=False, test_size=None, task=i+1,
        allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if args.scenario=="task" else None
    ) for i in range(args.tasks)]
    average_accs = sum(accs)/args.tasks
    # -print on screen
    if verbose:
        print("\n Accuracy of final model on test-set:")
        for i in range(args.tasks):
            print(" - {} {}: {:.4f}".format("For classes from task" if args.scenario=="class" else "Task",
                                            i + 1, accs[i]))
        print('=> Average accuracy over all {} {}: {:.4f}\n'.format(
            args.tasks*classes_per_task if args.scenario=="class" else args.tasks,
            "classes" if args.scenario=="class" else "tasks", average_accs
        ))
    # -write out to text file
    output_file = open("{}/acc-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_accs))
    output_file.close()


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------#
    #----- EVALUATION of GENERATOR -----#
    #-----------------------------------#

    if (utils.checkattr(args, 'feedback') or train_gen) and args.experiment=="CIFAR100" and args.scenario=="class":

        # Dataset and model to be used
        test_set = ConcatDataset(test_datasets)
        gen_model = model if utils.checkattr(args, 'feedback') else generator
        gen_model.eval()

        # Evaluate log-likelihood of generative model on combined test-set (with S=100 importance samples per datapoint)
        ll_per_datapoint = gen_model.estimate_loglikelihood(test_set, S=100, batch_size=args.batch)
        if verbose:
            print('=> Log-likelihood on test set: {:.4f} +/- {:.4f}\n'.format(
                np.mean(ll_per_datapoint), np.sqrt(np.var(ll_per_datapoint))
            ))
        # -write out to text file
        output_file = open("{}/ll-{}.txt".format(args.r_dir, param_stamp), 'w')
        output_file.write('{}\n'.format(np.mean(ll_per_datapoint)))
        output_file.close()

        # Evaluate reconstruction error (averaged over number of input units)
        re_per_datapoint = gen_model.calculate_recon_error(test_set, batch_size=args.batch, average=True)
        if verbose:
            print('=> Reconstruction error (per input unit) on test set: {:.4f} +/- {:.4f}\n'.format(
                np.mean(re_per_datapoint), np.sqrt(np.var(re_per_datapoint))
            ))
        # -write out to text file
        output_file = open("{}/re-{}.txt".format(args.r_dir, param_stamp), 'w')
        output_file.write('{}\n'.format(np.mean(re_per_datapoint)))
        output_file.close()

        # Try loading the classifier (our substitute for InceptionNet) for calculating IS, FID and Recall & Precision
        # -define model
        config['classes'] = 100
        pretrained_classifier = define.define_classifier(args=args, config=config, device=device)
        pretrained_classifier.hidden = False
        # -load pretrained weights
        eval_tag = "" if args.eval_tag=="none" else "-{}".format(args.eval_tag)
        try:
            utils.load_checkpoint(pretrained_classifier, args.m_dir, verbose=True,
                                  name="{}{}".format(pretrained_classifier.name, eval_tag))
            FileFound = True
        except FileNotFoundError:
            if verbose:
                print("= Could not find model {}{} in {}".format(pretrained_classifier.name, eval_tag, args.m_dir))
                print("= IS, FID and Precision & Recall not computed!")
            FileFound = False
        pretrained_classifier.eval()

        # Only continue with computing these measures if the requested classifier network (using --eval-tag) was found
        if FileFound:
            # Preparations
            total_n = len(test_set)
            n_repeats = int(np.ceil(total_n/args.batch))
            # -sample data from generator (for IS, FID and Precision & Recall)
            gen_x = gen_model.sample(size=total_n, only_x=True)
            # -generate predictions for generated data (for IS)
            gen_pred = []
            for i in range(n_repeats):
                x = gen_x[(i*args.batch): int(min(((i+1)*args.batch), total_n))]
                with torch.no_grad():
                    gen_pred.append(F.softmax(
                        pretrained_classifier.hidden_to_output(x) if args.hidden else pretrained_classifier(x), dim=1
                    ).cpu().numpy())
            gen_pred = np.concatenate(gen_pred)
            # -generate embeddings for generated data (for FID and Precision & Recall)
            gen_emb = []
            for i in range(n_repeats):
                with torch.no_grad():
                    gen_emb.append(pretrained_classifier.feature_extractor(
                        gen_x[(i*args.batch):int(min(((i+1)*args.batch), total_n))], from_hidden=args.hidden
                    ).cpu().numpy())
            gen_emb = np.concatenate(gen_emb)
            # -generate embeddings for test data (for FID and Precision & Recall)
            data_loader = utils.get_data_loader(test_set, batch_size=args.batch, cuda=cuda)
            real_emb = []
            for real_x, _ in data_loader:
                with torch.no_grad():
                    real_emb.append(pretrained_classifier.feature_extractor(real_x.to(device)).cpu().numpy())
            real_emb = np.concatenate(real_emb)

            # Calculate "Inception Score" (IS)
            py = gen_pred.mean(axis=0)
            is_per_datapoint = []
            for i in range(len(gen_pred)):
                pyx = gen_pred[i, :]
                is_per_datapoint.append(entropy(pyx, py))
            IS = np.exp(np.mean(is_per_datapoint))
            if verbose:
                print('=> Inception Score = {:.4f}\n'.format(IS))
            # -write out to text file
            output_file = open("{}/is{}-{}.txt".format(args.r_dir, eval_tag, param_stamp), 'w')
            output_file.write('{}\n'.format(IS))
            output_file.close()

            ## Calculate "Frechet Inception Distance" (FID)
            FID = fid.calculate_fid_from_embedding(gen_emb, real_emb)
            if verbose:
                print('=> Frechet Inception Distance = {:.4f}\n'.format(FID))
            # -write out to text file
            output_file = open("{}/fid{}-{}.txt".format(args.r_dir, eval_tag, param_stamp), 'w')
            output_file.write('{}\n'.format(FID))
            output_file.close()

            # Calculate "Precision & Recall"-curves
            precision, recall = pr.compute_prd_from_embedding(gen_emb, real_emb)
            # -write out to text files
            file_name = "{}/precision{}-{}.txt".format(args.r_dir, eval_tag, param_stamp)
            with open(file_name, 'w') as f:
                for item in precision:
                    f.write("%s\n" % item)
            file_name = "{}/recall{}-{}.txt".format(args.r_dir, eval_tag, param_stamp)
            with open(file_name, 'w') as f:
                for item in recall:
                    f.write("%s\n" % item)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # If requested, generate pdf
    if args.pdf:
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = evaluate.visual.plt.open_pdf(plot_name)

        # -show metrics reflecting progression during training
        if args.train and (not utils.checkattr(args, 'only_last')):
            # -create list to store all figures to be plotted.
            figure_list = []
            # -generate figures (and store them in [figure_list])
            figure = evaluate.visual.plt.plot_lines(
                progress_dict["all_tasks"], x_axes=[
                    i*classes_per_task for i in progress_dict["x_task"]
                ] if args.scenario=="class" else progress_dict["x_task"],
                line_names=['{} {}'.format(
                    "episode / task" if args.scenario=="class" else "task", i+1
                ) for i in range(args.tasks)],
                xlabel="# of {}s so far".format("classe" if args.scenario=="class" else "task"), ylabel="Test accuracy"
            )
            figure_list.append(figure)
            figure = evaluate.visual.plt.plot_lines(
                [progress_dict["average"]], x_axes=[
                    i*classes_per_task for i in progress_dict["x_task"]
                ] if args.scenario=="class" else progress_dict["x_task"],
                line_names=['Average based on all {}s so far'.format(
                    ("digit" if args.experiment=="splitMNIST" else "classe") if args.scenario else "task"
                )], xlabel="# of {}s so far".format("classe" if args.scenario=="class" else "task"),
                ylabel="Test accuracy"
            )
            figure_list.append(figure)
            # -add figures to pdf
            for figure in figure_list:
                pp.savefig(figure)

        gen_eval = (utils.checkattr(args, 'feedback') or train_gen)
        # -show samples (from main model or separate generator)
        if gen_eval and not no_samples:
            evaluate.show_samples(model if utils.checkattr(args, 'feedback') else generator, config, size=args.sample_n,
                                  pdf=pp, title="Generated samples (by final model)")

        # -plot "Precision & Recall"-curve
        if gen_eval and args.experiment=="CIFAR100" and args.scenario=="class" and FileFound:
            figure = evaluate.visual.plt.plot_pr_curves([[precision]], [[recall]])
            pp.savefig(figure)

        # -close pdf
        pp.close()

        # -print name of generated plot on screen
        if verbose:
            print("\nGenerated plot: {}\n".format(plot_name))



if __name__ == '__main__':
    args = handle_inputs()
    run(args, verbose=True)