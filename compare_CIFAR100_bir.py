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
    kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True, 'compare_code': 'bir'}
    # Define input options
    parser = options.define_args(filename="_compare_CIFAR100_bir",
                                 description='Compare different components of BI-R on split CIFAR-100.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    # Parse and process (i.e., set defaults for unselected options) options
    args = parser.parse_args()
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
        args.train = True
        main_cl.run(args)
    # -get average accuracies
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return tuple with the results
    return (ave, None)

def get_gen_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    eval_tag = "" if args.eval_tag=="none" else "-{}".format(args.eval_tag)
    if not os.path.isfile("{}/acc-{}.txt".format(args.r_dir, param_stamp)):
        print("{}: ...running...".format(param_stamp))
        args.train = True
        main_cl.run(args)
    elif (os.path.isfile("{}/ll-{}.txt".format(args.r_dir, param_stamp)) and
            os.path.isfile("{}/is{}-{}.txt".format(args.r_dir, eval_tag, param_stamp))):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running evaluation only...".format(param_stamp))
        args.train = False
        main_cl.run(args)
    # -get average accuracies
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -get log-likelihoods
    fileName = '{}/ll-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ll = float(file.readline())
    file.close()
    # -get reconstruction error (per input unit)
    fileName = '{}/re-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    re = float(file.readline())
    file.close()
    # -get inception score
    fileName = '{}/is{}-{}.txt'.format(args.r_dir, eval_tag, param_stamp)
    file = open(fileName)
    IS = float(file.readline())
    file.close()
    # -get Frechet inception distance
    fileName = '{}/fid{}-{}.txt'.format(args.r_dir, eval_tag, param_stamp)
    file = open(fileName)
    FID = float(file.readline())
    file.close()
    # -get precision and recall curve
    file_name = '{}/precision{}-{}.txt'.format(args.r_dir, eval_tag, param_stamp)
    precision = []
    with open(file_name, 'r') as f:
        for line in f:
            precision.append(float(line[:-1]))
    file_name = '{}/recall{}-{}.txt'.format(args.r_dir, eval_tag, param_stamp)
    recall = []
    with open(file_name, 'r') as f:
        for line in f:
            recall.append(float(line[:-1]))
    # -return tuple with the results
    return (ave, ll, re, IS, FID, precision, recall)

def collect_all(method_dict, seed_list, args, name=None, no_gen=False):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args) if no_gen else get_gen_results(args)
    # -return updated dictionary with results
    return method_dict

def barplots(dict1, dict2, names1, names2, ids, colors, seed_list, index, only_last_dir=None, chance_level=None,
             long_names1=None, long_names2=None, ylabel=None, title=None, ylim=None, perc=False, neg=False):

    # should results be multiplied by 100 and/or -1?
    multi = 100 if perc else 1
    multi = -multi if neg else multi

    # collect results
    dots1 = [[multi*dict1[id][seed][index] for seed in seed_list] for id in ids]
    means1 = [np.mean([multi*dict1[id][seed][index] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems1 = [np.sqrt(np.var([multi*dict1[id][seed][index] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    dots2 = [[multi*dict2[id][seed][index] for seed in seed_list] for id in ids]
    means2 = [np.mean([multi*dict2[id][seed][index] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems2 = [np.sqrt(np.var([multi*dict2[id][seed][index] for seed in seed_list])/(len(seed_list)-1)) for id in ids]

    # performance of base network only trained on last task
    if only_last_dir is not None:
        h_lines = [np.mean([multi*only_last_dir[seed][0] for seed in seed_list])]
        if len(seed_list)>1:
            h_errors = [np.sqrt(np.var([multi*only_last_dir[seed][0] for seed in seed_list])/(len(seed_list)-1))]
        else:
            h_errors = None
        h_labels = ["only trained on final task"]
        h_colors = ["black"]
    else:
        h_lines = h_errors = h_labels = h_colors = None

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

def pr_curves(result_dict, ids, seed_list, colors, names, i_recall, i_precision, title_top=None, title=None,
              alpha=None):

    # create [precision_list] and [recall_list] with <lists> of the precision- and recall-curves to be plotted
    precision_list = []
    recall_list = []
    # -loop over groups
    for i, id in enumerate(ids):
        # -create <list> to list all curves for this group
        precision_group = []
        recall_group = []
        # -loop over lines
        for seed in seed_list:
            precision_group.append(result_dict[id][seed][i_precision])
            recall_group.append(result_dict[id][seed][i_recall])
        # -add groups to master-list
        precision_list.append(precision_group)
        recall_list.append(recall_group)

    # create & return figure
    figure = plt.plot_pr_curves(precision_list, recall_list, names=names, colors=colors, with_dots=False,
                                linestyle="solid", title=title, title_top=title_top, alpha=alpha)
    return figure



if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    # --------------------------------------------------------------------------------#

    # Hard-coded, selected hyper-parameter values (obtained by running `./compare_CIFAR100_hyperParams --per-bir-comp`)
    if args.scenario=="class":
        dg_prop_bir = 0.7
        dg_prop_nD = 0.8
        dg_prop_nF = 0.7
        dg_prop_nC = 0.6
        dg_prop_nI = 0.9
        dg_prop_oG = 0.7
    elif args.scenario=="task":
        dg_prop_bir = 0.
        dg_prop_nD = 0.
        dg_prop_nF = 0.3
        dg_prop_nC = 0.
        dg_prop_nI = 0.8
        dg_prop_oG = 0.1

    # --------------------------------------------------------------------------------#

    ## Add default arguments
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False

    ## Use pre-trained convolutional layers for all compared methods
    args.pre_convE = True

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINE"----###

    ## Base-network only trained on last task / episode
    if not args.eval_gen:
        args.replay = "none"
        args.only_last = True
        ONLYLAST = {}
        ONLYLAST = collect_all(ONLYLAST, seed_list, args, name="Only last task", no_gen=True)
        args.only_last = False


    ###----"BI-R"----###
    args.replay = "generative"
    args.distill = True
    args.hidden = True
    args.prior = "GMM"
    args.per_class = True
    args.feedback = True
    args.dg_gates = True
    args.freeze_convE = True

    ## BI-R
    args.dg_prop = dg_prop_bir
    BIR = {}
    BIR = collect_all(BIR, seed_list, args, name="Brain-Inspired Replay (BI-R)", no_gen=(not args.eval_gen))


    ###----"Ablating BI-R components"----###

    ## No replay-through-feedback
    args.feedback = False
    args.dg_prop = dg_prop_nF
    BIR_nF = {}
    BIR_nF = collect_all(BIR_nF, seed_list, args, name="BI-R: no replay-through-feedback", no_gen=(not args.eval_gen))
    args.feedback = True

    ## No conditional replay
    args.prior = "standard"
    args.per_class = False
    args.dg_prop = dg_prop_nC
    BIR_nC = {}
    BIR_nC = collect_all(BIR_nC, seed_list, args, name="BI-R: no conditional replay", no_gen=(not args.eval_gen))
    args.prior = "GMM"
    args.per_class = True

    ## No gating based on internal context
    args.dg_gates = False
    args.dg_prop = 0.
    BIR_nG = {}
    BIR_nG = collect_all(BIR_nG, seed_list, args, name="BI-R: no gating based on internal context",
                         no_gen=(not args.eval_gen))
    args.dg_gates = True

    ## No internal replay
    args.hidden = False
    args.dg_prop = dg_prop_nI
    BIR_nI = {}
    BIR_nI = collect_all(BIR_nI, seed_list, args, name="BI-R: no internal replay", no_gen=(not args.eval_gen))
    args.hidden = True

    ## No distillation
    args.distill = False
    args.dg_prop = dg_prop_nD
    BIR_nD = {}
    BIR_nD = collect_all(BIR_nD, seed_list, args, name="BI-R: no distillation", no_gen=(not args.eval_gen))
    args.distill = True


    ###----"Standard GR"----###
    args.replay = "generative"
    args.distill = False
    args.hidden = False
    args.prior = "standard"
    args.per_class = False
    args.feedback = False
    args.dg_gates = False
    args.freeze_convE = False

    ## Standard GR
    SGR = {}
    SGR = collect_all(SGR, seed_list, args, name="Standard GR (s-GR))", no_gen=(not args.eval_gen))


    ###----"Adding BI-R components"----###

    ## With replay-through-feedback
    args.feedback = True
    SGR_wF = {}
    SGR_wF = collect_all(SGR_wF, seed_list, args, name="s-GR: with replay-through-feedback", no_gen=(not args.eval_gen))
    args.feedback = False

    ## With conditional replay
    args.per_class = True
    args.prior = "GMM"
    SGR_wC = {}
    SGR_wC = collect_all(SGR_wC, seed_list, args, name="s-GR: with conditional replay", no_gen=(not args.eval_gen))
    args.per_class = False
    args.prior = "standard"

    ## With gating based on internal context
    args.dg_gates = True
    args.dg_prop = dg_prop_oG
    SGR_wG = {}
    SGR_wG = collect_all(SGR_wG, seed_list, args, name="s-GR: with gating based on internal context", no_gen=(not args.eval_gen))
    args.dg_gates = False

    ## With internal replay
    args.hidden = True
    args.freeze_convE = True
    SGR_wI = {}
    SGR_wI = collect_all(SGR_wI, seed_list, args, name="s-GR: with internal replay", no_gen=(not args.eval_gen))
    args.hidden = False
    args.freeze_convE = False

    ## With distillation
    args.distill = True
    SGR_wD = {}
    SGR_wD = collect_all(SGR_wD, seed_list, args, name="s-GR: with distillation", no_gen=(not args.eval_gen))
    args.distill = False


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    ALL_bir = [BIR, BIR_nF, BIR_nC, BIR_nG, BIR_nI, BIR_nD]
    ALL_sg = [SGR, SGR_wF, SGR_wC, SGR_wG, SGR_wI, SGR_wD]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # print header to screen
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"#"*60)

    # open pdf
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    pp = plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # select names / colors / ids
    names_bir = ["BI-R", "- rtf", "- con", "- gat", "- int", "- dis"]
    long_names_bir = ["BI-R", "- replay-through-feedback", "- conditional replay", "- gating", "- internal replay",
                      "- distillation"]
    names_sg = ["s-GR", "+ rtf", "+ con", "+ gat", "+ int", "+ dis"]
    long_names_sg = ["s-GR", "+ replay-through-feedback", "+ conditional replay", "+ gating", "+ internal replay",
                     "+ distillation"]
    colors = ["black", "maroon", "red", "orangered", "goldenrod", "green"]
    ids = [0,1,2,3,4,5]

    # total number of classes to be learned
    if args.scenario=="class":
        total_classes = args.tasks*int(np.floor(100/args.tasks))

    if not args.eval_gen:
        ##--- AVERAGE ACCURACY ---##
        index = 0
        ylabel = "Test accuracy (after all {} {})".format(args.tasks, "episodes" if args.scenario=="class" else "tasks")
        title = "AVERAGE ACCURACY (in %)"
        chance_level = (100./total_classes) if args.scenario=="class" else (100./int(np.floor(100/args.tasks)))
        print("\n\n{}\n".format(title)+"-"*60)
        figure = barplots(ALL_sg, ALL_bir, names_sg, names_bir, ids, colors, seed_list, index,
                          only_last_dir=ONLYLAST, chance_level=chance_level,
                          long_names1=long_names_sg, long_names2=long_names_bir,
                          ylabel=ylabel, title=title, ylim=None, perc=True, neg=False)
        figure_list.append(figure)

    else:
        ##--- LOG LIKELIHOOD ---##
        index = 1
        ylabel = "Average test negative log-likelihood (over all {})".format(
            "{} classes".format(total_classes) if args.scenario=="class" else "{} tasks".format(args.tasks)
        )
        title = "NEGATIVE LOG-LIKELIHOOD"
        print("\n\n{}\n".format(title)+"-"*60)
        figure = barplots(ALL_sg, ALL_bir, names_sg, names_bir, ids, colors, seed_list, index,
                          long_names1=long_names_sg, long_names2=long_names_bir,
                          ylabel=ylabel, title=title, ylim=None, perc=False, neg=True)
        figure_list.append(figure)

        ##--- RECONSTRUCTION ERROR ---##
        index = 2
        ylabel = "Average test reconstruction error (MSE), divided by number of units (over all {})".format(
            "{} classes".format(total_classes) if args.scenario=="class" else "{} tasks".format(args.tasks)
        )
        title = "RECONSTRUCTION ERROR (PER INPUT UNIT)"
        print("\n\n{}\n".format(title)+"-"*60)
        figure = barplots(ALL_sg, ALL_bir, names_sg, names_bir, ids, colors, seed_list, index,
                          long_names1=long_names_sg, long_names2=long_names_bir,
                          ylabel=ylabel, title=title, ylim=None, perc=False, neg=False)
        figure_list.append(figure)

        ##--- INCEPTION SCORE ---##
        index = 3
        ylabel = "Modified IS (after all {} {})".format(args.tasks, "episodes" if args.scenario=="class" else "tasks")
        title = "'INCEPTION' SCORE"
        print("\n\n{}\n".format(title)+"-"*60)
        figure = barplots(ALL_sg, ALL_bir, names_sg, names_bir, ids, colors, seed_list, index,
                          long_names1=long_names_sg, long_names2=long_names_bir,
                          ylabel=ylabel, title=title, ylim=None, perc=False, neg=False)
        figure_list.append(figure)

        ##--- FRECHET INCEPTION DISTANCE ---##
        index = 4
        ylabel = "Modified FID (after all {} {})".format(args.tasks, "episodes" if args.scenario=="class" else "tasks")
        title = "FRECHET 'INCEPTION' DISTANCE"
        print("\n\n{}\n".format(title)+"-"*60)
        figure = barplots(ALL_sg, ALL_bir, names_sg, names_bir, ids, colors, seed_list, index,
                          long_names1=long_names_sg, long_names2=long_names_bir,
                          ylabel=ylabel, title=title, ylim=None, perc=False, neg=False)
        figure_list.append(figure)

        ##--- PRECISION & RECALL ---##
        title = "PRECISION-RECALL CURVES"
        fig = pr_curves(ALL_sg, ids, seed_list, colors, names_sg, i_recall=6, i_precision=5, title_top=title,
                        title="Additions to Standard GR", alpha=0.7)
        figure_list.append(fig)
        fig = pr_curves(ALL_bir, ids, seed_list, colors, names_bir, i_recall=6, i_precision=5, title_top=title,
                        title="Ablations from BI-R")
        figure_list.append(fig)


    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))