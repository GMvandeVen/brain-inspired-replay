import argparse
from utils import checkattr


##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


## Option-indicators to be passed to each of the below functions:
    # single_task:      <bool>
    # only_MNIST:       <bool>
    # generative:       <bool>
    # compare_code:     none | all | hyper | replay | bir


def add_general_options(parser, single_task=False, generative=False, compare_code="none", only_MNIST=True, **kwargs):
    parser.add_argument('--no-save', action='store_false', dest='save', help="don't save trained models")
    if single_task and generative:
            parser.add_argument('--save-all', action='store_true', help="also store conv- and deconv-layers")
    if not only_MNIST:
        parser.add_argument('--convE-stag', type=str, metavar='STAG', default='none',help="tag for saving convE-layers")
    parser.add_argument('--full-stag', type=str, metavar='STAG', default='none', help="tag for saving full model")
    parser.add_argument('--full-ltag', type=str, metavar='LTAG', default='none', help="tag for loading full model")
    parser.add_argument('--test', action='store_false', dest='train', help='evaluate previously saved model')
    if not single_task:
        parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
    if compare_code in ("none"):
        parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
    else:
        parser.add_argument('--seed', type=int, default=11, help='[first] random seed (for each random-module used)')
        parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
    parser.add_argument('--data-dir', type=str, default='./store/datasets', dest='d_dir', help="default: %(default)s")
    parser.add_argument('--model-dir', type=str, default='./store/models', dest='m_dir', help="default: %(default)s")
    parser.add_argument('--plot-dir', type=str, default='./store/plots', dest='p_dir', help="default: %(default)s")
    if not single_task:
        parser.add_argument('--results-dir', type=str, default='./store/results', dest='r_dir',
                            help="default: %(default)s")
    return parser


def add_eval_options(parser, single_task=False, generative=False, compare_code="none", only_MNIST=True, **kwargs):
    # evaluation parameters
    eval = parser.add_argument_group('Evaluation Parameters')
    eval.add_argument('--pdf', action='store_true', help="generate pdf with plots for individual experiment(s)")
    eval.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
    if compare_code=="none" and not single_task:
        eval.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
    if compare_code=="none":
        eval.add_argument('--loss-log', type=int, default=500, metavar="N", help="# iters after which to plot loss")
        eval.add_argument('--acc-log', type=int, default=None if single_task else 500, metavar="N",
                          help="# iters after which to plot accuracy")
    eval.add_argument('--acc-n', type=int, default=1024, help="# samples for evaluating accuracy (visdom-plots)")
    if compare_code=="none" and generative:
            eval.add_argument('--sample-log', type=int, default=1000, metavar="N",
                              help="# iters after which to plot samples")
    if generative:
        eval.add_argument('--sample-n', type=int, default=64, help="# images to show")
        eval.add_argument('--no-samples', action='store_true', help="don't plot generated/reconstructed images")
        if not only_MNIST:
            eval.add_argument('--eval-tag', type=str, metavar="ETAG", default="none", help="tag for evaluation model")
        if (not only_MNIST) and compare_code=="bir":
            eval.add_argument('--eval-gen', action='store_true',
                              help="instead of accuracy, evaluate quality of generators")
    return parser


def add_task_options(parser, only_MNIST=False, single_task=False, compare_code="none", **kwargs):
    # expirimental task parameters
    task_params = parser.add_argument_group('Task Parameters')
    if single_task:
        task_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'MNIST28']
        task_default = 'CIFAR10'
    else:
        MNIST_tasks = ['splitMNIST', 'permMNIST']
        image_tasks = ['CIFAR100']
        task_choices = MNIST_tasks if only_MNIST else MNIST_tasks+image_tasks
        task_default = 'splitMNIST' if only_MNIST else 'CIFAR100'
    task_params.add_argument('--experiment', type=str, default=task_default, choices=task_choices)
    if not single_task:
        task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
        # 'task':   each task has own output-units, always only those units are considered
        # 'domain': each task is mapped to the same output-units
        # 'class':  each task has own output-units, all units of tasks seen so far are considered
        task_params.add_argument('--tasks', type=int, help='number of tasks')
    if not only_MNIST:
        task_params.add_argument('--augment', action='store_true',
                                 help="augment training data (random crop & horizontal flip)")
        task_params.add_argument('--no-norm', action='store_false', dest='normalize',
                                 help="don't normalize images (only for CIFAR)")
    if not single_task and compare_code=="none":
        task_params.add_argument('--only-last', action='store_true', help="only train on last task / episode")
    return parser


def add_permutedMNIST_task_options(parser, **kwargs):
    # expirimental task parameters specific for the permuted MNIST protocol
    task_params = parser.add_argument_group('Task Parameters')
    task_params.add_argument('--tasks', type=int, help='number of permutations')
    return parser


def add_model_options(parser, only_MNIST=False, generative=False, single_task=False, **kwargs):
    # model architecture parameters
    model = parser.add_argument_group('Parameters Main Model')
    if not only_MNIST:
        # -conv-layers
        model.add_argument('--conv-type', type=str, default="standard", choices=["standard", "resNet"])
        model.add_argument('--n-blocks', type=int, default=2, help="# blocks per conv-layer (only for 'resNet')")
        model.add_argument('--depth', type=int, default=5 if single_task else None,
                           help="# of convolutional layers (0 = only fc-layers)")
        model.add_argument('--reducing-layers', type=int, dest='rl',help="# of layers with stride (=image-size halved)")
        model.add_argument('--channels', type=int, default=16, help="# of channels 1st conv-layer (doubled every 'rl')")
        model.add_argument('--conv-bn', type=str, default="yes", help="use batch-norm in the conv-layers (yes|no)")
        model.add_argument('--conv-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
        model.add_argument('--global-pooling', action='store_true', dest='gp', help="ave global pool after conv-layers")
    # -fully-connected-layers
    model.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, default=2000 if single_task else None, metavar="N",
                       help="# of units in first fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])
    model.add_argument('--h-dim', type=int, metavar="N", help='# of hidden units final layer (default: fc-units)')
    # NOTE: number of units per fc-layer linearly declinces from [fc_units] to [h_dim].
    if generative:
        model.add_argument('--z-dim', type=int, default=100,help='size of latent representation (if feedback, def=100)')
    return parser


def add_train_options(parser, only_MNIST=False, single_task=False, generative=False, compare_code="none", **kwargs):
    # training hyperparameters / initialization
    train_params = parser.add_argument_group('Training Parameters')
    if single_task:
        iter_epochs = train_params.add_mutually_exclusive_group(required=False)
        iter_epochs.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='max # of epochs (default: %(default)d)')
        iter_epochs.add_argument('--iters', type=int, metavar='N', help='max # of iterations (replaces "--epochs")')
    else:
        train_params.add_argument('--iters', type=int, help="# batches to optimize main model")
    train_params.add_argument('--lr', type=float, default=0.0001 if single_task else None, help="learning rate")
    train_params.add_argument('--batch', type=int, default=256 if single_task else None, help="batch-size")
    train_params.add_argument('--init-weight', type=str, default='standard', choices=['standard', 'xavier'])
    train_params.add_argument('--init-bias', type=str, default='standard', choices=['standard', 'constant'])
    if not single_task and compare_code not in ('replay'):
        train_params.add_argument('--reinit', action='store_true', help='reinitialize networks before each new task')
    if not only_MNIST:
        if compare_code in ("none"):
            train_params.add_argument('--pre-convE', action='store_true', help="use pretrained convE-layers")
        train_params.add_argument('--convE-ltag', type=str, metavar='LTAG', default='none',
                                  help="tag for loading convE-layers")
        if compare_code in ("none") and generative:
                train_params.add_argument('--pre-convD', action='store_true', help="use pretrained convD-layers")
        if compare_code in ("none"):
            train_params.add_argument('--freeze-convE', action='store_true', help="freeze parameters of convE-layers")
        if compare_code in ("none") and generative:
            train_params.add_argument('--freeze-convD', action='store_true', help="freeze parameters of convD-layers")
    if generative:
        train_params.add_argument('--recon-loss', type=str, choices=['MSE', 'BCE'])
    return parser


def add_VAE_options(parser, only_MNIST=False,  **kwargs):
    VAE = parser.add_argument_group('VAE-specific Parameters')
    # -how to weigh components of the loss-function?
    VAE.add_argument('--recon-weight', type=float, default=1., dest='rcl', help="weight of recon-loss (def=1)")
    VAE.add_argument('--variat-weight', type=float, default=1., dest='vl', help="weight of KLD-loss (def=1)")
    # -architecture of decoder (type of deconv-layer and use of 'decoder-gates')
    if not only_MNIST:
        VAE.add_argument('--deconv-type', type=str, default="standard", choices=["standard", "resNet"])
    return parser


def add_replay_options(parser, only_MNIST=False, compare_code="none", **kwargs):
    replay = parser.add_argument_group('Replay Parameters')
    if compare_code in ("none"):
        replay_choices = ['offline', 'generative', 'none', 'current']
        replay.add_argument('--replay', type=str, default='none', choices=replay_choices)
    if compare_code not in ("all", "hyper", "bir"):
        replay.add_argument('--distill', action='store_true', help="use distillation for replay")
    replay.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    if compare_code not in ('replay'):
        replay.add_argument('--batch-replay', type=int, metavar='N', help="batch-size for replay (default: batch)")
    # - generative model parameters (only if separate generator)
    if not only_MNIST:
        replay.add_argument('--g-depth', type=int, help='[depth] in generator (default: same as classifier)')
    replay.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
    if compare_code not in ('replay'):
        replay.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
        replay.add_argument('--g-h-dim', type=int, help='[h_dim] in generator (default: same as classifier)')
    replay.add_argument('--g-z-dim', type=int, default=100, help="size of generator's latent representation (def=100)")
    # - hyper-parameters (again only if separate generator)
    replay.add_argument('--gen-iters', type=int, dest="g_iters", help="# batches to optimize generator (def=[iters])")
    replay.add_argument('--lr-gen', type=float, help="learning rate (separate) generator (default: lr)")
    # -add VAE-specific parameters
    if compare_code in ("none"):
        parser = add_VAE_options(parser, only_MNIST=only_MNIST)
    return parser


def add_bir_options(parser, only_MNIST=False, compare_code="none", **kwargs):
    BIR = parser.add_argument_group('Brain-inspired Replay Parameters')
    # -use all default options for brain-inspired replay
    if compare_code in ("none"):
        BIR.add_argument('--brain-inspired', action='store_true', help="select defaults for brain-inspired replay")
    # -feedback-related parameters
    if compare_code in ("none"):
        BIR.add_argument('--feedback', action="store_true", help="equip main model with feedback connections")
    BIR.add_argument('--pred-weight', type=float, default=1., dest='pl', help="(FB) weight of prediction loss (def=1)")
    # -where on the VAE should the softmax-layer be placed?
    BIR.add_argument('--classify', type=str, default="beforeZ", choices=['beforeZ', 'fromZ'])
    # -prior-related parameters
    if compare_code in ("none"):
        BIR.add_argument('--prior', type=str, default="standard", choices=["standard", "GMM"])
        BIR.add_argument('--per-class', action='store_true', help="if selected, each class has own modes")
    BIR.add_argument('--n-modes', type=int, default=1, help="how many modes for prior (per class)? (def=1)")
    # -decoder-gate-related parameters
    if compare_code in ('none'):
        BIR.add_argument('--dg-gates', action='store_true', help="use context-specific gates in decoder")
    BIR.add_argument('--dg-type', type=str, metavar="TYPE", help="decoder-gates: based on tasks or classes?")
    if not compare_code in ('hyper', 'bir'):
        BIR.add_argument('--dg-prop', type=float, help="decoder-gates: masking-prop")
    if compare_code in ('all'):
        BIR.add_argument('--dg-si-prop', type=float, metavar="PROP", help="decoder-gates: masking-prop for BI-R + SI")
        BIR.add_argument('--dg-c', type=float, metavar="C", help="SI hyperparameter for BI-R + SI")
    # -hidden replay
    if (not only_MNIST) and compare_code in ("none"):
        BIR.add_argument('--hidden', action="store_true", help="replay at 'internal level' (after conv-layers)")
    return parser


def add_allocation_options(parser, compare_code="none", **kwargs):
    cl = parser.add_argument_group('Memory Allocation Parameters')
    if compare_code in ("none"):
        cl.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
    if not compare_code in ('hyper'):
        cl.add_argument('--lambda', type=float, dest="ewc_lambda",help="--> EWC: regularisation strength")
    if compare_code in ("none"):
        cl.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
    elif compare_code in ("all"):
        cl.add_argument('--o-lambda', type=float, help="--> online EWC: regularisation strength")
    if not compare_code in ('hyper'):
        cl.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
    cl.add_argument('--fisher-n', type=int, default=1000, help="--> EWC: sample size estimating Fisher Information")
    if compare_code in ("none"):
        cl.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
    if not compare_code in ('hyper'):
        cl.add_argument('--c', type=float, dest="si_c", help="-->  SI: regularisation strength")
    cl.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="-->  SI: dampening parameter")
    if compare_code in ('none'):
        cl.add_argument('--xdg', action='store_true', help="use 'Context-dependent Gating' (Masse et al, 2018)")
    if not compare_code in ('hyper'):
        cl.add_argument('--xdg-prop', type=float, dest='xdg_prop', help="--> XdG: prop neurons per layer to gate")
    return  parser


##-------------------------------------------------------------------------------------------------------------------##

############################
## Check / modify options ##
############################

def set_defaults(args, only_MNIST=False, single_task=False, generative=True, compare_code='None', **kwargs):
    # -if 'brain-inspired' is selected, select corresponding defaults
    if checkattr(args, 'brain_inspired'):
        if hasattr(args, "replay") and not args.replay=="generative":
            raise Warning("To run with brain-inspired replay, select both '--brain-inspired' and '--replay=generative'")
        args.feedback = True     #--> replay-through-feedback
        args.prior = 'GMM'       #--> conditional replay
        args.per_class = True    #--> conditional replay
        args.dg_gates = True     #--> gating based on internal context (has hyper-param 'dg_prop')
        args.hidden = True       #--> internal replay
        args.pre_convE = True    #--> internal replay
        args.freeze_convE = True #--> internal replay
        args.distill = True      #--> distillation
    # -set default-values for certain arguments based on chosen experiment
    args.normalize = args.normalize if args.experiment in ('CIFAR10', 'CIFAR100') else False
    args.augment = args.augment if args.experiment in ('CIFAR10', 'CIFAR100') else False
    if hasattr(args, "depth"):
        args.depth = (5 if args.experiment in ('CIFAR10', 'CIFAR100') else 0) if args.depth is None else args.depth
    if hasattr(args, "recon_loss"):
        args.recon_loss = (
            "MSE" if args.experiment in ('CIFAR10', 'CIFAR100') else "BCE"
        ) if args.recon_loss is None else args.recon_loss
    if hasattr(args, "dg_type"):
        args.dg_type = ("task" if args.experiment=='permMNIST' else "class") if args.dg_type is None else args.dg_type
    if not single_task:
        args.tasks= (
            5 if args.experiment=='splitMNIST' else (10 if args.experiment=="CIFAR100" else 100)
        ) if args.tasks is None else args.tasks
        args.iters = (5000 if args.experiment=='CIFAR100' else 2000) if args.iters is None else args.iters
        args.lr = (0.001 if args.experiment=='splitMNIST' else 0.0001) if args.lr is None else args.lr
        args.batch = (128 if args.experiment=='splitMNIST' else 256) if args.batch is None else args.batch
        args.fc_units = (400 if args.experiment=='splitMNIST' else 2000) if args.fc_units is None else args.fc_units
    # -set hyper-parameter values (typically found by grid-search) based on chosen experiment & scenario
    if not single_task and not compare_code in ('hyper', 'bir'):
        if args.experiment=='splitMNIST':
            args.xdg_prop = 0.9 if args.scenario=="task" and args.xdg_prop is None else args.xdg_prop
            args.si_c = (10. if args.scenario=='task' else 0.1) if args.si_c is None else args.si_c
            args.ewc_lambda = (
                1000000000. if args.scenario=='task' else 100000.
            ) if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1. if args.gamma is None else args.gamma
            if hasattr(args, 'dg_prop'):
                args.dg_prop = 0.8 if args.dg_prop is None else args.dg_prop
        elif args.experiment=='CIFAR100':
            args.xdg_prop = 0.7 if args.scenario=="task" and args.xdg_prop is None else args.xdg_prop
            args.si_c = (100. if args.scenario=='task' else 1.) if args.si_c is None else args.si_c
            args.ewc_lambda = (1000. if args.scenario=='task' else 1.) if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1 if args.gamma is None else args.gamma
            args.dg_prop = (0. if args.scenario == "task" else 0.7) if args.dg_prop is None else args.dg_prop
            if compare_code=="all":
                args.dg_si_prop = 0.6 if args.dg_si_prop is None else args.dg_si_prop
                args.dg_c = 100000000. if args.dg_c is None else args.dg_c
        elif args.experiment=='permMNIST':
            args.si_c = 10. if args.si_c is None else args.si_c
            args.ewc_lambda = 1. if args.ewc_lambda is None else args.ewc_lambda
            if hasattr(args, 'o_lambda'):
                args.o_lambda = 1. if args.o_lambda is None else args.o_lambda
            args.gamma = 1. if args.gamma is None else args.gamma
            args.dg_prop = 0.8 if args.dg_prop is None else args.dg_prop
            if compare_code=="all":
                args.dg_si_prop = 0.8 if args.dg_si_prop is None else args.dg_si_prop
                args.dg_c = 1. if args.dg_c is None else args.dg_c
    # -for other unselected options, set default values (not specific to chosen scenario / experiment)
    args.h_dim = args.fc_units if args.h_dim is None else args.h_dim
    if hasattr(args, "lr_gen"):
        args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    if hasattr(args, "rl"):
        args.rl = args.depth-1 if args.rl is None else args.rl
    if generative and not single_task:
        if hasattr(args, 'g_iters'):
            args.g_iters = args.iters if args.g_iters is None else args.g_iters
        if hasattr(args, 'g_depth') and not only_MNIST:
            args.g_depth = args.depth if args.g_depth is None else args.g_depth
        if hasattr(args, 'g_fc_lay'):
            args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
        if hasattr(args, 'g_fc_uni'):
            args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
        if hasattr(args, "g_h_dim"):
            args.g_h_dim = args.g_fc_uni if args.g_h_dim is None else args.g_h_dim
    if (not single_task) and (not compare_code in ('hyper')):
        args.xdg_prop = 0. if args.scenario=="task" and args.xdg_prop is None else args.xdg_prop
    # -if [log_per_task] (which is default for comparison-scripts), reset all logs
    if not single_task:
        args.log_per_task = True if (not compare_code=="none") else args.log_per_task
    if checkattr(args, 'log_per_task'):
        args.acc_log = args.iters
        args.loss_log = args.iters
        args.sample_log = args.iters
    return args


def check_for_errors(args, single_task=False, **kwargs):
    # -errors in scenario-specification
    if not single_task:
        # -if scenario is "class" and XdG is selected, give error
        if args.scenario=="class" and checkattr(args, 'xdg') and args.xdg_prop>0:
            raise ValueError("Having scenario=[class] with 'XdG' does not make sense")
        # -if scenario is "domain" and XdG is selected, give warning
        if args.scenario=="domain" and checkattr(args, 'xdg') and args.xdg_prop>0:
            print("Although scenario=[domain], 'XdG' makes that task identity is nevertheless always required")
        # -if XdG is selected together with replay of any kind, give error
        if checkattr(args, 'xdg') and args.xdg_prop>0 and (not args.replay=="none"):
            raise NotImplementedError("XdG is not supported with '{}' replay.".format(args.replay))
            #--> problem is that applying different task-masks interferes with gradient calculation
            #    (should be possible to overcome by calculating each gradient before applying next mask)
        # -if 'only_last' is selected with replay, EWC or SI, give error
        if checkattr(args, 'only_last') and (not args.replay=="none"):
            raise NotImplementedError("Option 'only_last' is not supported with '{}' replay.".format(args.replay))
        if checkattr(args, 'only_last') and (checkattr(args, 'ewc') and args.ewc_lambda>0):
            raise NotImplementedError("Option 'only_last' is not supported with EWC.")
        if checkattr(args, 'only_last') and (checkattr(args, 'si') and args.si_c>0):
            raise NotImplementedError("Option 'only_last' is not supported with SI.")
    # -error in type of reconstruction loss
    if checkattr(args, "normalize") and hasattr(args, "recon_los") and args.recon_loss=="BCE":
        raise ValueError("'BCE' is not a valid reconstruction loss with normalized images")