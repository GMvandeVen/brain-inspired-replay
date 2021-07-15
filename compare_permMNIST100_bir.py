#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import options
import utils
from visual import plt
import main_cl


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': True, 'generative': True, 'compare_code': 'bir'}
    # Define input options
    parser = options.define_args(filename="_compare_CIFAR100_bir",
                                 description='Compare different components of BI-R on permuted MNIST.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_permutedMNIST_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    # Parse and process (i.e., set defaults for unselected options) options
    args = parser.parse_args()
    args.scenario = "domain"
    args.experiment = "permMNIST"
    args.xdg_prop = 0
    options.set_defaults(args, **kwargs)
    return args


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile("{}/acc-{}.txt".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main_cl.run(args)
    # -get average accuracies
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return
    return ave


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


def barplots(result_list1, result_list2, names1, names2, ids, colors, seed_list, only_last_dir, chance_level=None,
             long_names1=None, long_names2=None, ylabel=None, title=None, ylim=None, perc=False, neg=False):

    # should results be multiplied by 100 and/or -1?
    multi = 100 if perc else 1
    multi = -multi if neg else multi

    # collect results
    dots1 = [[multi*result_list1[seed][id] for seed in seed_list] for id in ids]
    means1 = [np.mean([multi*result_list1[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems1 = [np.sqrt(np.var([multi*result_list1[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    dots2 = [[multi * result_list2[seed][id] for seed in seed_list] for id in ids]
    means2 = [np.mean([multi*result_list2[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems2 = [np.sqrt(np.var([multi*result_list2[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]

    # performance of base network only trained on last task
    h_lines = [np.mean([multi*only_last_dir[seed] for seed in seed_list])]
    if len(seed_list)>1:
        h_errors = [np.sqrt(np.var([multi*only_last_dir[seed] for seed in seed_list])/(len(seed_list)-1))]
    else:
        h_errors = None
    h_labels = ["only trained on final task"]
    h_colors = ["black"]

    # bar-plot
    figure = plt.plot_bars([means1, means2], names=[names1, names2], colors=[colors, colors], ylabel=ylabel,
                           yerr=[sems1, sems2] if len(seed_list)>1 else None, ylim=ylim, top_title=title,
                           title_list=["Additions to Standard GR", "Ablations from BI-R"], vlines=[0.5, 0.5],
                           alpha=[0.7, 1], dots=[dots1, dots2] if len(seed_list)>1 else None,
                           h_line=chance_level, h_label="chance level" if chance_level is not None else None,
                           h_lines=h_lines, h_errors=h_errors, h_labels=h_labels, h_colors=h_colors)

    # print results to screen
    for i,name in enumerate(long_names1 if long_names1 is not None else names1):
        if len(seed_list) > 1:
            print("{:26s} {:9.2f}  (+/- {:6.2f}),  n={}".format(name, means1[i], sems1[i], len(seed_list)))
        else:
            print("{:30s} {:9.2f}".format(name, means1[i]))
    print("-"*60)
    for i,name in enumerate(long_names2 if long_names2 is not None else names2):
        if len(seed_list) > 1:
            print("{:26s} {:9.2f}  (+/- {:6.2f}),  n={}".format(name, means2[i], sems2[i], len(seed_list)))
        else:
            print("{:30s} {:9.2f}".format(name, means2[i]))
    print("-"*60)

    return(figure)



if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    # --------------------------------------------------------------------------------#

    # Selected hyper-parameter values (obtained by running `./compare_permMNIST100_hyperParams --per-bir-comp`)
    dg_prop_bir = 0.8
    dg_prop_nF = 0.8
    dg_prop_nC = 0.
    dg_prop_nD = 0.6
    dg_prop_oG = 0.2

    # --------------------------------------------------------------------------------#

    ## Add default arguments (some of which will be different for different runs)
    args.replay = "none"
    args.distill = False
    args.feedback = False
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## Base-network only trained on last permutation
    args.only_last = True
    ONLYLAST = {}
    ONLYLAST = collect_all(ONLYLAST, seed_list, args, name="Only last task")
    args.only_last = False


    ###----"BI-R"----###
    args.replay = "generative"
    args.distill = True
    args.prior = "GMM"
    args.per_class = True
    args.feedback = True
    args.dg_gates = True

    ## BI-R
    args.dg_prop = dg_prop_bir
    BIR = {}
    BIR = collect_all(BIR, seed_list, args, name="Brain-Inspired Replay (BI-R)")


    ###----"Ablating BI-R components"----###

    ## No feedback
    args.feedback = False
    args.dg_prop = dg_prop_nF
    BIR_nF = {}
    BIR_nF = collect_all(BIR_nF, seed_list, args, name="BI-R: no replay-through-feedback")
    args.feedback = True

    ## No conditional replay
    args.prior = "standard"
    args.per_class = False
    args.dg_prop = dg_prop_nC
    BIR_nC = {}
    BIR_nC = collect_all(BIR_nC, seed_list, args, name="BI-R: no conditional replay")
    args.prior = "GMM"
    args.per_class = True

    ## No gating
    args.dg_gates = False
    args.dg_prop = 0.
    BIR_nG = {}
    BIR_nG = collect_all(BIR_nG, seed_list, args, name="BI-R: no gating based on internal context")
    args.dg_gates = True

    ## No distillation
    args.distill = False
    args.dg_prop = dg_prop_nD
    BIR_nD = {}
    BIR_nD = collect_all(BIR_nD, seed_list, args, name="BI-R: no distillation")
    args.distill = True


    ###----"Standard GR"----###
    args.replay = "generative"
    args.distill = False
    args.prior = "standard"
    args.per_class = False
    args.feedback = False
    args.dg_gates = False

    ## Standard GR
    SGR = {}
    SGR = collect_all(SGR, seed_list, args, name="Standard GR (s-GR))")


    ###----"Adding BI-R components"----###

    ## With feedback
    args.feedback = True
    SGR_wF = {}
    SGR_wF = collect_all(SGR_wF, seed_list, args, name="s-GR: with feedback")
    args.feedback = False

    ## With conditional replay
    args.per_class = True
    args.prior = "GMM"
    SGR_wC = {}
    SGR_wC = collect_all(SGR_wC, seed_list, args, name="s-GR: with conditional replay")
    args.per_class = False
    args.prior = "standard"

    ## With context gates
    args.dg_gates = True
    args.dg_prop = dg_prop_oG
    SGR_wG = {}
    SGR_wG = collect_all(SGR_wG, seed_list, args, name="s-GR: with context gates")
    args.dg_gates = False

    ## With distillation
    args.distill = True
    SGR_wD = {}
    SGR_wD = collect_all(SGR_wD, seed_list, args, name="s-GR: with distillation")
    args.distill = False


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    ave_acc_bir = {}
    ave_acc_sg = {}

    ## Create lists for all extracted <dicts> and <lists> with fixed order
    for seed in seed_list:
        ave_acc_bir[seed] = [BIR[seed], BIR_nF[seed], BIR_nC[seed], BIR_nG[seed], BIR_nD[seed]]
        ave_acc_sg[seed] = [SGR[seed], SGR_wF[seed], SGR_wC[seed], SGR_wG[seed], SGR_wD[seed]]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # print header to screen
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    print("\n\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"#"*60)

    # open pdf
    plot_name = "birPerComp-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    pp = plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # select names / colors / ids
    names_bir = ["BI-R", "- rtf", "- con", "- gat", "- dis"]
    long_names_bir = ["BI-R", "- replay-through-feedback", "- conditional replay", "- gating", "- distillation"]
    names_sg = ["s-GR", "+ rtf", "+ con", "+ gat", "+ dis"]
    long_names_sg = ["s-GR", "+ replay-through-feedback", "+ conditional replay", "+ gating", "+ distillation"]
    colors = ["black", "maroon", "red", "orangered", "green"]
    ids = [0,1,2,3,4]

    ##--- AVERAGE ACCURACY ---##
    ylabel = "Average test accuracy (over all {} permutations)".format(args.tasks)
    title = "AVERAGE TEST ACCURACY (in %)"
    print("\n{}\n".format(title)+"-"*60)
    figure = barplots(ave_acc_sg, ave_acc_bir, names_sg, names_bir, ids, colors, seed_list, only_last_dir=ONLYLAST,
                      chance_level=10, long_names1=long_names_sg, long_names2=long_names_bir,
                      ylabel=ylabel, title=title, ylim=None, perc=True)
    figure_list.append(figure)

    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))