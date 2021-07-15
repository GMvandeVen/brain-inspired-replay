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
    parser = options.define_args(filename="_compare_MNIST",
                                 description='Compare performance of various continual learning strategies on different'
                                             ' scenarios of splitMNIST.')
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

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## None
    args.replay = "none"
    args.distill = False
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")

    ## Joint
    args.replay = "offline"
    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Joint")


    ###----"REPLAY"----###

    ## Generative replay
    args.replay = "generative"
    GR = {}
    GR = collect_all(GR, seed_list, args, name="Generative Replay (GR)")

    ## LwF
    args.replay = "current"
    args.distill = True
    LWF = {}
    LWF = collect_all(LWF, seed_list, args, name="Learning without Forgetting (LwF)")
    args.replay = "none"
    args.distill = False


    ###----"EWC / SI"----####

    ## EWC
    args.ewc = True
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="Elastic Weight Consolidation (EWC)")
    args.ewc = False

    ## SI
    args.si = True
    SI = {}
    SI = collect_all(SI, seed_list, args, name="Synaptic Intelligence (SI)")
    args.si = False


    ###----"XdG"----####
    if args.scenario=="task":
        args.xdg = True
        XDG = {}
        XDG = collect_all(XDG, seed_list, args, name="Context-dependent Gating (XdG)")
        args.xdg = False


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
            LWF[seed][i]["average"], GR[seed][i]["average"], EWC[seed][i]["average"], SI[seed][i]["average"],
        ]
        i = 1
        ave_acc[seed] = [
            OFF[seed][i], NONE[seed][i], LWF[seed][i], GR[seed][i], EWC[seed][i], SI[seed][i],
        ]
        if args.scenario=="task":
            ave_acc[seed].append(XDG[seed][1])
            acc[seed].append(XDG[seed][0]["average"])


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "{}-incremental learning".format(args.scenario)
    title = "{}   -   {}".format(args.experiment, scheme)
    ylabel_all = "Test accuracy ({})".format("based on all digits" if args.scenario=="class" else "after all tasks")
    ylabel = "Test accucary ({} so far)".format("based on digits" if args.scenario=="class" else "on tasks")
    x_axes = NONE[args.seed][0]["x_task"]

    # select names / colors / ids
    if args.scenario=="task":
        names = ["None", "EWC", "SI", "XdG", "LwF", "GR", "Joint" ]
        colors = ["grey", "darkgreen", "yellowgreen", "deepskyblue", "goldenrod", "red", "black"]
        ids = [1, 4, 5, 6, 2, 3, 0]
    else:
        names = ["None", "EWC", "SI", "LwF", "GR", "Joint" ]
        colors = ["grey", "darkgreen", "yellowgreen", "goldenrod", "red", "black"]
        ids = [1, 4, 5, 2, 3, 0]

    # open pdf
    pp = plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # bar-plot
    means = [np.mean([ave_acc[seed][id] for seed in seed_list]) for id in ids]
    if args.n_seeds>1:
        sems = [np.sqrt(np.var([ave_acc[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = plt.plot_bar(means, names=names, colors=colors, ylabel=ylabel_all, title=title,
                          yerr=sems if args.n_seeds>1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:19s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:19s} {:.2f}".format(name, 100*means[i]))
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
    ylim = (0,1.02) if args.scenario=="class" else None
    figure = plt.plot_lines(ave_lines, x_axes=[2*i for i in x_axes] if args.scenario=="class" else x_axes,
                            line_names=names, colors=colors, title=title,
                            xlabel="# {} so far".format("digits" if args.scenario=="class" else "tasks"), ylabel=ylabel,
                            list_with_errors=sem_lines if args.n_seeds>1 else None, ylim=ylim)
    figure_list.append(figure)

    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))