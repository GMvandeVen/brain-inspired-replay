#!/usr/bin/env python3
import os
import numpy as np
import options
from param_stamp import get_param_stamp_from_args
import utils
from visual import plt
import main_cl



## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': True, 'generative': True, 'compare_code': 'all'}
    # Define input options
    parser = options.define_args(filename="_compare_permMNIST100",
                                 description='Compare performance of "continual learning strategies" on long instance '
                                             'of standard version (i.e., Domain-IL) of permMNIST (100 tasks).')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_permutedMNIST_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Parse and process (i.e., set defaults for unselected options) options
    args = parser.parse_args()
    args.scenario = "domain"
    args.experiment = "permMNIST"
    options.set_defaults(args, **kwargs)
    return args



def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile("{}/dict-{}.pkl".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main_cl.run(args)
    # -get average accuracies
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -results-dict
    dict = utils.load_object("{}/dict-{}".format(args.r_dir,  param_stamp))
    # -return tuple with the results
    return (dict, ave)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict



if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    ## Store gating proportion for decoder-gates
    gating_prop = args.dg_prop
    args.dg_prop = 0

    ## Add default arguments (might be different for different runs)
    args.replay = "none"
    args.distill = False
    args.feedback = False
    args.prior = "standard"
    args.per_class = False
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = 0.

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## Joint
    args.replay = "offline"
    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Joint")

    ## None
    args.replay = "none"
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")


    ###----"COMPETING METHODS"----###

    ## LwF
    args.replay = "current"
    args.distill = True
    LWF = {}
    LWF = collect_all(LWF, seed_list, args, name="LwF")
    args.replay = "none"
    args.distill = False

    ## SI
    args.si = True
    SI = {}
    SI = collect_all(SI, seed_list, args, name="SI")
    args.si = False

    ## EWC
    args.ewc = True
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")
    args.ewc = False

    ## online EWC
    args.ewc = True
    args.online = True
    args.ewc_lambda = args.o_lambda
    OEWC = {}
    OEWC = collect_all(OEWC, seed_list, args, name="Online EWC")
    args.ewc = False
    args.online = False


    ###----"REPLAY VARIANTS"----###

    ## GR
    args.replay = "generative"
    args.prior = "standard"
    args.per_class = False
    args.feedback = False
    args.distill = False
    GR = {}
    GR = collect_all(GR, seed_list, args, name="GR")

    ## BI-R
    args.prior = "GMM"
    args.per_class = True
    args.feedback = True
    args.dg_gates = True
    args.dg_prop = gating_prop
    args.distill = True
    BIR = {}
    BIR = collect_all(BIR, seed_list, args, name="Brain-Inspired Replay (BI-R)")

    ## BI-R & SI
    args.si = True
    args.dg_prop = args.dg_si_prop
    args.si_c = args.dg_c
    BIRpSI = {}
    BIRpSI = collect_all(BIRpSI, seed_list, args, name="BI-R + SI")


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    acc = {}
    ave_acc = {}

    ## Create lists for all extracted <dicts> and <lists> with fixed order
    for seed in seed_list:

        i = 0
        acc[seed] = [
            OFF[seed][i]["average"], NONE[seed][i]["average"],
            LWF[seed][i]["average"], GR[seed][i]["average"],
            EWC[seed][i]["average"], OEWC[seed][i]["average"], SI[seed][i]["average"],
            BIR[seed][i]["average"], BIRpSI[seed][i]["average"],
        ]
        i = 1
        ave_acc[seed] = [
            OFF[seed][i], NONE[seed][i],
            LWF[seed][i], GR[seed][i],
            EWC[seed][i], OEWC[seed][i], SI[seed][i],
            BIR[seed][i], BIRpSI[seed][i],
        ]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    x_axes = GR[args.seed][0]["x_task"]

    # select names / colors / ids
    names = ["None", "LwF", "EWC", "Online EWC", "SI", "Generative Replay (GR)", "Brain-Inspired Replay (BI-R)",
             "BI-R + SI", "Joint"]
    colors = ["grey", "goldenrod", "darkgreen", "green", "yellowgreen", "red", "purple", "blue", "black"]
    ids = [1,2,4,5,6,3,7,8,0]

    # open pdf
    pp = plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # bar-plot
    means = [np.mean([ave_acc[seed][id] for seed in seed_list]) for id in ids]
    if args.n_seeds>1:
        sems = [np.sqrt(np.var([ave_acc[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = plt.plot_bar(means, names=names, colors=colors, ylabel="Test accuracy (after all 100 tasks)", title=title,
                          yerr=sems if args.n_seeds>1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:30s} {:5.2f}  (+/- {:4.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:34s} {:5.2f}".format(name, 100*means[i]))
    print("#"*60)

    # line-plot
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(acc[args.seed][id])):
            all_entries = [acc[seed][id][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if args.n_seeds>1:
                new_sem_line.append(np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    ylim = (0.3,1)
    figure = plt.plot_lines(ave_lines, x_axes=x_axes, line_names=names, colors=colors, title=title,
                            xlabel="# of permutations so far", ylabel="Test accuracy (on tasks seen so far)",
                            list_with_errors=sem_lines if args.n_seeds>1 else None, ylim=ylim)
    figure_list.append(figure)


    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))