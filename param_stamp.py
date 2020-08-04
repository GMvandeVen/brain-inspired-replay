from data.load import get_multitask_experiment
from utils import checkattr


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''
    from define_models import define_autoencoder, define_classifier

    # -get configurations of experiment
    config = get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir, only_config=True,
        normalize=args.normalize if hasattr(args, "normalize") else False, verbose=False,
    )

    # -get model architectures
    model = define_autoencoder(args=args, config=config, device='cpu') if checkattr(
        args,'feedback'
    ) else define_classifier(args=args, config=config, device='cpu')
    if checkattr(args, 'feedback'):
        model.lamda_pl = 1. if not hasattr(args, 'pl') else args.pl
    train_gen = (hasattr(args, 'replay') and args.replay=="generative" and not checkattr(args, 'feedback'))
    if train_gen:
        generator = define_autoencoder(args=args, config=config, device='cpu', generator=True,
                                       convE=model.convE if hasattr(args, "hidden") and args.hidden else None)

    # -extract and return param-stamp
    model_name = model.name
    replay_model_name = generator.name if train_gen else None
    param_stamp = get_param_stamp(args, model_name, replay=(hasattr(args, "replay") and not args.replay=="none"),
                                  replay_model_name=replay_model_name, verbose=False)
    return param_stamp



def get_param_stamp(args, model_name, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}-{set}{of}".format(
        n=args.tasks, set=args.scenario, of="OL" if checkattr(args, 'only_last') else ""
    ) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{norm}{aug}{multi_n}".format(
        exp=args.experiment, norm="-N" if hasattr(args, 'normalize') and args.normalize else "",
        aug="+" if hasattr(args, "augment") and args.augment else "", multi_n=multi_n_stamp
    )
    if verbose:
        print(" --> task:          "+task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for hyper-parameters
    pre_conv = ""
    if (checkattr(args, "pre_convE") or checkattr(args, "pre_convD")) and (hasattr(args, 'depth') and args.depth>0):
        ltag = "" if not hasattr(args, "convE_ltag") or args.convE_ltag=="none" else "-{}".format(args.convE_ltag)
        pre_conv = "-pCvE{}".format(ltag) if args.pre_convE else "-pCvD"
        pre_conv = "-pConv{}".format(ltag) if args.pre_convE and checkattr(args, "pre_convD") else pre_conv
    freeze_conv = ""
    if (checkattr(args, "freeze_convD") or checkattr(args, "freeze_convE")) and hasattr(args, 'depth') and args.depth>0:
        freeze_conv = "-fCvE" if checkattr(args, "freeze_convE") else "-fCvD"
        freeze_conv = "-fConv" if checkattr(args, "freeze_convE") and checkattr(args, "freeze_convD") else freeze_conv
    hyper_stamp = "{i_e}{num}-lr{lr}{lrg}-b{bsz}{pretr}{freeze}{reinit}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        lrg=("" if args.lr==args.lr_gen else "-lrG{}".format(args.lr_gen)) if (
            hasattr(args, "lr_gen") and hasattr(args, "replay") and args.replay=="generative" and
            (not checkattr(args, "feedback"))
        ) else "",
        bsz=args.batch, pretr=pre_conv, freeze=freeze_conv, reinit="-R" if checkattr(args, 'reinit') else ""
    )
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)

    # -for EWC / SI
    if (checkattr(args, 'ewc') and args.ewc_lambda>0) or (checkattr(args, 'si') and args.si_c>0):
        ewc_stamp = "EWC{l}-{fi}{o}".format(
            l=args.ewc_lambda, fi="{}".format("N" if args.fisher_n is None else args.fisher_n),
            o="-O{}".format(args.gamma) if checkattr(args, 'online') else "",
        ) if (checkattr(args, 'ewc') and args.ewc_lambda>0) else ""
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.epsilon) if (checkattr(args,'si') and args.si_c>0) else ""
        both = "--" if (checkattr(args,'ewc') and args.ewc_lambda>0) and (checkattr(args,'si') and args.si_c>0) else ""
        if verbose and checkattr(args, 'ewc') and args.ewc_lambda>0:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and checkattr(args, 'si') and args.si_c>0:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}".format(ewc_stamp, both, si_stamp) if (
            (checkattr(args, 'ewc') and args.ewc_lambda>0) or (checkattr(args, 'si') and args.si_c>0)
    ) else ""

    # -for XdG
    xdg_stamp = ""
    if (checkattr(args, "xdg") and args.xdg_prop > 0):
        xdg_stamp = "--XdG{}".format(args.xdg_prop)
        if verbose:
            print(" --> XdG:           " + "gating = {}".format(args.xdg_prop))

    # -for replay
    if replay:
        replay_stamp = "{H}{rep}{bat}{distil}{model}{gi}".format(
            H="" if not args.replay=="generative" else (
                "H" if (checkattr(args, "hidden") and hasattr(args, 'depth') and args.depth>0) else ""
            ),
            rep="gen" if args.replay=="generative" else args.replay,
            bat="" if (
                    (not hasattr(args, 'batch_replay')) or (args.batch_replay is None) or args.batch_replay==args.batch
            ) else "-br{}".format(args.batch_replay),
            distil="-Di{}".format(args.temp) if args.distill else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
            gi="-gi{}".format(args.g_iters) if (
                hasattr(args, "g_iters") and (replay_model_name is not None) and (not args.iters==args.g_iters)
            ) else "",
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""

    # -for choices regarding reconstruction loss
    if checkattr(args, "feedback"):
        recon_stamp = "--{}{}".format(
            "H_" if checkattr(args, "hidden") and hasattr(args, 'depth') and args.depth>0 else "", args.recon_loss
        )
    elif hasattr(args, "replay") and args.replay=="generative":
        recon_stamp = "--{}".format(args.recon_loss)
    else:
        recon_stamp = ""

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, ewc_stamp, xdg_stamp, replay_stamp,
        recon_stamp, "-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp