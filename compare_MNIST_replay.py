#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import options
from visual import plt
import main_cl


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': True, 'generative': True, 'compare_code': 'replay'}
    # Define input options
    parser = options.define_args(filename="_compare_MNIST_replay",
                                 description="Generative replay: effect of quantity & quality.")
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Parse and process (i.e., set defaults for unselected options) options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    return args



## Parameter-values to compare
batch_r_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
g_fc_uni_list = [10, 20, 40, 100, 200, 400, 1000]



def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/acc-{}.txt'.format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main_cl.run(args)
    # -get average accuracy
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return it
    return ave


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_result(args)
    # -return updated dictionary with results
    return method_dict



if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    ## Add default arguments (will be different for different runs)
    args.replay = "generatie"
    args.reinit = False
    args.batch_replay = None
    args.g_fc_uni = args.fc_units
    args.g_h_dim = args.h_dim
    # args.seed will also vary!

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###### GENERATIVE REPLAY VARIANTS #########
    args.replay = "generative"

    ## GR with batch-size
    GR_b = {}
    for batch_r in batch_r_list:
        args.batch_replay = batch_r
        GR_b[batch_r] = {}
        GR_b[batch_r] = collect_all(GR_b[batch_r], seed_list, args,
                                    name="GR - batch-size replay = {}".format(batch_r))

    ## GR+reinit with batch-size
    GR_br = {}
    args.reinit = True
    for batch_r in batch_r_list:
        args.batch_replay = batch_r
        GR_br[batch_r] = {}
        GR_br[batch_r] = collect_all(GR_br[batch_r], seed_list, args,
                                     name="GR & reinit - batch-size replay = {}".format(batch_r))
    args.reinit = False
    args.batch_replay = None

    ## GR with gen-size
    GR_g = {}
    for g_fc in g_fc_uni_list:
        args.g_fc_uni = g_fc
        args.g_h_dim = g_fc
        GR_g[g_fc] = {}
        GR_g[g_fc] = collect_all(GR_g[g_fc], seed_list, args,
                                 name="GR - # units per hidden layer VAE = {}".format(g_fc))

    ## GR+reinit with gen-size
    GR_gr = {}
    args.reinit = True
    for g_fc in g_fc_uni_list:
        args.g_fc_uni = g_fc
        args.g_h_dim = g_fc
        GR_gr[g_fc] = {}
        GR_gr[g_fc] = collect_all(GR_gr[g_fc], seed_list, args,
                                  name="GR & reinit - # units per hidden layer VAE = {}".format(g_fc))
    args.reinit = False



    ###### OTHER APPROACHES #########

    ## None
    args.replay = "none"
    args.distill = False
    BASE = {}
    BASE = collect_all(BASE, seed_list, args, name="None")

    ## Joint
    args.replay = "offline"
    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Joint")

    ## EWC
    args.replay = "none"
    args.ewc = True
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")
    args.ewc = False

    ## SI
    args.si = True
    SI = {}
    SI = collect_all(SI, seed_list, args, name="SI")
    args.si = False

    ## LwF
    args.replay = "current"
    args.distill = True
    LWF = {}
    LWF = collect_all(LWF, seed_list, args, name="LwF")
    args.replay = "none"
    args.distill = False

    ## XdG
    if args.scenario=="task":
        args.xdg = True
        XDG = {}
        XDG = collect_all(XDG, seed_list, args, name="XdG")
        args.xdg = False



    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summaryReplay-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)

    # open pdf
    pp = plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # set scale of y-axis
    y_lim = [0,1] if args.scenario=="class" else None

    # methods for comparison
    h_lines = [
        np.mean([BASE[seed] for seed in seed_list]),
        np.mean([EWC[seed] for seed in seed_list]),
        np.mean([SI[seed] for seed in seed_list]),
        np.mean([LWF[seed] for seed in seed_list]),
        np.mean([(XDG[seed] if args.scenario=="task" else OFF[seed]) for seed in seed_list])
    ]
    if args.scenario=="task":
        h_lines.append(np.mean([OFF[seed] for seed in seed_list]))
    h_errors = [
        np.sqrt(np.var([BASE[seed] for seed in seed_list]) / (len(seed_list)-1)),
        np.sqrt(np.var([EWC[seed] for seed in seed_list]) / (len(seed_list) - 1)),
        np.sqrt(np.var([SI[seed] for seed in seed_list]) / (len(seed_list) - 1)),
        np.sqrt(np.var([LWF[seed] for seed in seed_list]) / (len(seed_list) - 1)),
        np.sqrt(np.var([(XDG[seed] if args.scenario=="task" else OFF[seed]) for seed in seed_list]) / (len(seed_list) - 1)),
    ] if args.n_seeds>1 else None
    if args.scenario=="task" and args.n_seeds>1:
        h_errors.append(np.sqrt(np.var([OFF[seed] for seed in seed_list]) / (len(seed_list) - 1)))
    h_labels = ["None", "EWC", "SI", "LwF", "XdG" if args.scenario=="task" else "Joint"]
    h_colors = ["grey", "darkgreen", "yellowgreen", "goldenrod", "deepskyblue" if args.scenario=="task" else "black"]
    if args.scenario=="task":
        h_labels.append("Joint")
        h_colors.append("black")

    # names & colors of main lines
    line_names = ["GR", "GR + reinit"]
    colors = ["red", "saddlebrown"]

    # graph comparing replay quantity
    ave_GR = []
    sem_GR = []
    ave_GRr = []
    sem_GRr = []
    for batch_r in batch_r_list:
        all_entries = [GR_b[batch_r][seed] for seed in seed_list]
        ave_GR.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_GR.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

        all_entries = [GR_br[batch_r][seed] for seed in seed_list]
        ave_GRr.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_GRr.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

    averages = [ave_GR, ave_GRr]
    if args.n_seeds>1:
        sems = [sem_GR, sem_GRr]

    figure = plt.plot_lines_with_baselines(
        averages,
        x_axes=batch_r_list, ylabel="Test accuracy (after all tasks)", title=title, x_log=True, ylim=y_lim,
        line_names=line_names, xlabel="Replay batch-size (log-scale)", with_dots=True,
        list_with_errors=sems if args.n_seeds > 1 else None,
        h_lines=h_lines, h_errors=h_errors, h_labels=h_labels, h_colors=h_colors, colors=colors
    )
    figure_list.append(figure)

    # graph comparing replay quality
    ave_GR = []
    sem_GR = []
    ave_GRr = []
    sem_GRr = []
    for g_fc in g_fc_uni_list:
        all_entries = [GR_g[g_fc][seed] for seed in seed_list]
        ave_GR.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_GR.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

        all_entries = [GR_gr[g_fc][seed] for seed in seed_list]
        ave_GRr.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_GRr.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

    averages = [ave_GR, ave_GRr]
    if args.n_seeds>1:
        sems = [sem_GR, sem_GRr]

    figure = plt.plot_lines_with_baselines(
        averages,
        x_axes=g_fc_uni_list, ylabel="Test accuracy (after all tasks)", title=title, x_log=True, ylim=y_lim,
        line_names=line_names, xlabel="# of units in hidden layers VAE (log-scale)", with_dots=True,
        list_with_errors=sems if args.n_seeds > 1 else None,
        h_lines=h_lines, h_errors=h_errors, h_labels=h_labels, h_colors=h_colors, colors=colors
    )
    figure_list.append(figure)

    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))