#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import options
from visual import plt as my_plt
from matplotlib.pyplot import get_cmap
import main_cl



## Parameter-values to compare
lamda_list = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
              100000000000., 1000000000000.]
gamma_list = [1.]
c_list_TaskIL = [0.001, 0.01, 0.1, 1., 10. , 100., 1000., 10000., 100000., 1000000., 10000000., 100000000.]
c_list_ClassIL = [0.1, 1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000.,
                  10000000000.]
dg_prop_list = [0., 0.2, 0.4, 0.6, 0.8]
dg_prop_list_onlybir = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
xdg_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True, 'compare_code': 'hyper'}
    # Define input options
    parser = options.define_args(filename="_compare_CIFAR100_hyperParams",
                                 description='Compare hyperparameters fo split CIFAR-100.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    parser.add_argument('--per-bir-comp', action='store_true', help="also do gridsearch for individual BI-R components")
    # Parse and process (i.e., set defaults for unselected options) options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    return args


def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/acc-{}.txt'.format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main_cl.run(args)
    # -get average accuracies
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return it
    return ave


if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    ## Add default arguments (will be different for different runs)
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False
    args.freeze_convE = False

    ## Use pre-trained convolutional layers for all compared methods
    args.pre_convE = True

    ## Select list for SI based on chosen scenario
    c_list = c_list_TaskIL if args.scenario=="task" else c_list_ClassIL

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    ## Baselline
    BASE = get_result(args)

    ## EWC
    EWC = {}
    args.ewc = True
    for ewc_lambda in lamda_list:
        args.ewc_lambda=ewc_lambda
        EWC[ewc_lambda] = get_result(args)
    args.ewc = False

    ## Online EWC
    OEWC = {}
    args.ewc = True
    args.online = True
    for gamma in gamma_list:
        OEWC[gamma] = {}
        args.gamma = gamma
        for ewc_lambda in lamda_list:
            args.ewc_lambda = ewc_lambda
            OEWC[gamma][ewc_lambda] = get_result(args)
    args.ewc = False
    args.online = False

    ## SI
    SI = {}
    args.si = True
    for si_c in c_list:
        args.si_c = si_c
        SI[si_c] = get_result(args)
    args.si = False

    ## XdG
    if args.scenario=="task":
        XDG = {}
        args.xdg = True
        for xdg in xdg_list:
            args.xdg_prop = xdg
            XDG[xdg] = get_result(args)
        args.xdg = False

    ## Brain-inspired Replay
    BIR = {}
    args.replay = "generative"
    args.hidden = True
    args.distill = True
    args.prior = "GMM"
    args.per_class = True
    args.dg_gates = True
    args.feedback = True
    args.freeze_convE = True
    for dg_prop in dg_prop_list_onlybir:
        args.dg_prop = dg_prop
        BIR[dg_prop] = get_result(args)

    ## Brain-inspired Replay with SI
    BIR_SI = {}
    for dg_prop in dg_prop_list:
        BIR_SI[dg_prop] = {}
        args.dg_prop = dg_prop
        for si_c in [0]+c_list:
            args.si_c = si_c
            args.si = True if si_c>0 else False
            BIR_SI[dg_prop][si_c] = get_result(args)


    ## If requested, also perform gridsearch for addition- and ablation-experiments
    if args.per_bir_comp:
        args.si = False

        ## BI-R without replay-through-feedback
        BIR_no_RTF = {}
        args.feedback = False
        for dg_prop in dg_prop_list_onlybir:
            args.dg_prop = dg_prop
            BIR_no_RTF[dg_prop] = get_result(args)
        args.feedback = True

        ## BI-R without conditional replay
        BIR_no_CON = {}
        args.prior = "standard"
        args.per_class = False
        for dg_prop in dg_prop_list_onlybir:
            args.dg_prop = dg_prop
            BIR_no_CON[dg_prop] = get_result(args)
        args.prior = "GMM"
        args.per_class = True

        ## BI-R without internal replay
        BIR_no_INT = {}
        args.hidden = False
        for dg_prop in dg_prop_list_onlybir:
            args.dg_prop = dg_prop
            BIR_no_INT[dg_prop] = get_result(args)
        args.hidden = True

        ## BI-R without distillation
        BIR_no_DIS = {}
        args.distill = False
        for dg_prop in dg_prop_list_onlybir:
            args.dg_prop = dg_prop
            BIR_no_DIS[dg_prop] = get_result(args)
        args.distill = False


        ## Standard Generative Replay (GR) plus gating based on internal context
        GR_plus_GAT = {}
        args.distill = False
        args.prior = "standard"
        args.per_class = False
        args.feedback = False
        args.hidden = False
        args.dg_gates = True
        args.freeze_convE = False
        for dg_prop in dg_prop_list_onlybir:
            args.dg_prop = dg_prop
            GR_plus_GAT[dg_prop] = get_result(args)



    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------#
    #----- COLLECT DATA AND PRINT ON SCREEN -----#
    #--------------------------------------------#

    ext_c_list = [0] + c_list
    ext_lambda_list = [0] + lamda_list
    ext_xdg_list = [0] + xdg_list
    print("\n")


    ###---EWC + online EWC---###

    # -collect data
    ave_acc_ewc = [BASE] + [EWC[ewc_lambda] for ewc_lambda in lamda_list]
    ave_acc_per_lambda = [ave_acc_ewc]
    for gamma in gamma_list:
        ave_acc_temp = [BASE] + [OEWC[gamma][ewc_lambda] for ewc_lambda in lamda_list]
        ave_acc_per_lambda.append(ave_acc_temp)
    # -print on screen
    print("\n\nELASTIC WEIGHT CONSOLIDATION (EWC)")
    print(" param-list (lambda): {}".format(ext_lambda_list))
    print("  {}".format(ave_acc_ewc))
    print("--->  lambda = {}     --    {}".format(ext_lambda_list[np.argmax(ave_acc_ewc)], np.max(ave_acc_ewc)))
    if len(gamma_list) > 0:
        print("\n\nONLINE EWC")
        print(" param-list (lambda): {}".format(ext_lambda_list))
        curr_max = 0
        for gamma in gamma_list:
            ave_acc_temp = [BASE] + [OEWC[gamma][ewc_lambda] for ewc_lambda in lamda_list]
            print("  (gamma={}):   {}".format(gamma, ave_acc_temp))
            if np.max(ave_acc_temp) > curr_max:
                gamam_max = gamma
                lamda_max = ext_lambda_list[np.argmax(ave_acc_temp)]
                curr_max = np.max(ave_acc_temp)
        print("--->  gamma = {}  -  lambda = {}     --    {}".format(gamam_max, lamda_max, curr_max))


    ###---SI---###

    # -collect data
    ave_acc_si = [BASE] + [SI[c] for c in c_list]
    # -print on screen
    print("\n\nSYNAPTIC INTELLIGENCE (SI)")
    print(" param list (si_c): {}".format(ext_c_list))
    print("  {}".format(ave_acc_si))
    print("---> si_c = {}     --    {}".format(ext_c_list[np.argmax(ave_acc_si)], np.max(ave_acc_si)))


    ###---XdG---###

    if args.scenario == "task":
        # -collect data
        ave_acc_xdg = [BASE] + [XDG[c] for c in xdg_list]
        # -print on screen
        print("\n\nCONTEXT-DEPENDENT GATING (XDG))")
        print(" param list (gating_prop): {}".format(ext_xdg_list))
        print("  {}".format(ave_acc_xdg))
        print("---> gating_prop = {}     --    {}".format(ext_xdg_list[np.argmax(ave_acc_xdg)], np.max(ave_acc_xdg)))
    print('\n')


    ###---Brain-Inspired Replay (BI-R)---###

    # -collect data
    ave_acc_bir = [BIR[dg_prop] for dg_prop in dg_prop_list_onlybir]
    # -print on screen
    print("\n\nBRAIN-INSPIRED REPLAY (BI-R)")
    print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
    print("  {}".format(ave_acc_bir))
    print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir)], np.max(ave_acc_bir)))


    ###---BI-R + SI---###

    # -collect data
    ave_acc_bir_per_c = []
    for dg_prop in dg_prop_list:
        ave_acc_bir_per_c.append([BIR_SI[dg_prop][c] for c in ext_c_list])
    # -print on screen
    print("\n\nBI-R & SI")
    print(" param-list (si_c): {}".format(ext_c_list))
    curr_max = 0
    for dg_prop in dg_prop_list:
        ave_acc_temp = [BIR_SI[dg_prop][c] for c in ext_c_list]
        print("  (dg-prop={}):   {}".format(dg_prop, ave_acc_temp))
        if np.max(ave_acc_temp)>curr_max:
            dg_prop_max = dg_prop
            si_max = ext_c_list[np.argmax(ave_acc_temp)]
            curr_max = np.max(ave_acc_temp)
    print("--->  dg_prop = {}  -  si_c = {}     --    {}".format(dg_prop_max, si_max, curr_max))


    if args.per_bir_comp:

        ###---BI-R per component---###

        # -collect data
        ave_acc_bir_no_rtf = [BIR_no_RTF[dg_prop] for dg_prop in dg_prop_list_onlybir]
        ave_acc_bir_no_con = [BIR_no_CON[dg_prop] for dg_prop in dg_prop_list_onlybir]
        ave_acc_bir_no_int = [BIR_no_INT[dg_prop] for dg_prop in dg_prop_list_onlybir]
        ave_acc_bir_no_dis = [BIR_no_DIS[dg_prop] for dg_prop in dg_prop_list_onlybir]
        ave_acc_gr_plus_gat = [GR_plus_GAT[dg_prop] for dg_prop in dg_prop_list_onlybir]
        # -print on screen
        print("\n\nBI-R minus REPLAY-THROUGH-FEEDBACK")
        print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
        print("  {}".format(ave_acc_bir_no_rtf))
        print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir_no_rtf)],
                                                       np.max(ave_acc_bir_no_rtf)))
        print("\n\nBI-R minus CONDITIONAL REPLAY")
        print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
        print("  {}".format(ave_acc_bir_no_con))
        print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir_no_con)],
                                                       np.max(ave_acc_bir_no_con)))
        print("\n\nBI-R minus INTERNAL REPLAY")
        print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
        print("  {}".format(ave_acc_bir_no_int))
        print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir_no_int)],
                                                       np.max(ave_acc_bir_no_int)))
        print("\n\nBI-R minus DISTILLATION")
        print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
        print("  {}".format(ave_acc_bir_no_dis))
        print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir_no_dis)],
                                                       np.max(ave_acc_bir_no_dis)))
        print("\n\nGR plus GATING BASED ON INTERNAL CONTEXT")
        print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
        print("  {}".format(ave_acc_gr_plus_gat))
        print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_gr_plus_gat)],
                                                       np.max(ave_acc_gr_plus_gat)))
    print('\n')



    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "hyperParams-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    ylabel = "Average accuracy (after all tasks)"

    # calculate limits y-axes (to have equal for all graphs)
    full_list = [item for sublist in ave_acc_per_lambda for item in sublist] + ave_acc_si + ave_acc_bir + \
                [item for sublist in ave_acc_bir_per_c for item in sublist]
    if args.scenario=="task":
        full_list += ave_acc_xdg
    if args.per_bir_comp:
        full_list += (ave_acc_bir_no_rtf + ave_acc_bir_no_con + ave_acc_bir_no_dis + ave_acc_bir_no_int
                      + ave_acc_gr_plus_gat)
    miny = np.min(full_list)
    maxy = np.max(full_list)
    marginy = 0.1*(maxy-miny)

    # open pdf
    pp = my_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []


    ###---EWC + online EWC---###
    # - select colors
    colors = ["darkgreen"]
    colors += get_cmap('Greens')(np.linspace(0.7, 0.3, len(gamma_list))).tolist()
    # - make plot (line plot - only average)
    figure = my_plt.plot_lines(ave_acc_per_lambda, x_axes=ext_lambda_list, ylabel=ylabel,
                               line_names=["EWC"] + ["Online EWC - gamma = {}".format(gamma) for gamma in gamma_list],
                               title=title, x_log=True, xlabel="EWC: lambda (log-scale)",
                               ylim=(miny-marginy, maxy+marginy),
                               with_dots=True, colors=colors, h_line=BASE, h_label="None")
    figure_list.append(figure)


    ###---SI---###
    figure = my_plt.plot_lines([ave_acc_si], x_axes=ext_c_list, ylabel=ylabel, line_names=["SI"],
                            colors=["yellowgreen"], title=title, x_log=True, xlabel="SI: c (log-scale)", with_dots=True,
                            ylim=(miny-marginy, maxy+marginy), h_line=BASE, h_label="None")
    figure_list.append(figure)


    ###---XdG---###
    if args.scenario=="task":
        figure = my_plt.plot_lines([ave_acc_xdg], x_axes=ext_xdg_list, ylabel=ylabel,
                                line_names=["XdG"], colors=["deepskyblue"], ylim=(miny-marginy, maxy+marginy),
                                title=title, x_log=False, xlabel="XdG: % of nodes gated",
                                with_dots=True, h_line=BASE, h_label="None")
        figure_list.append(figure)


    ###---Brain-Inspired Replay---###
    figure = my_plt.plot_lines([ave_acc_bir], x_axes=dg_prop_list_onlybir, ylabel=ylabel,
                               line_names=["Brain-Inspired Replay (BI-R)"],
                               colors=["purple"], title=title, x_log=False, xlabel="Context gates: % of nodes gated",
                               with_dots=True, ylim=(miny-marginy, maxy+marginy), h_lines=[BASE], h_labels=["None"],
                               h_colors=["grey"])
    figure_list.append(figure)


    ###---Brain-Inspired Replay + SI---###
    # - select colors
    colors = get_cmap('Blues_r')(np.linspace(0.6, 0., len(dg_prop_list))).tolist()
    # - make plot (line plot - only average)
    figure = my_plt.plot_lines(ave_acc_bir_per_c, x_axes=ext_c_list, ylabel=ylabel,
                               line_names=["BI-R, gate-prop = {}".format(dg_prop) for dg_prop in dg_prop_list],
                               title=title, x_log=True, xlabel="BI-R + SI: c (log-scale)",
                               ylim=(miny-marginy, maxy+marginy),
                               with_dots=True, colors=colors, h_line=BASE, h_label="None")
    figure_list.append(figure)


    ###---BI-R per component---###
    if args.per_bir_comp:
        figure = my_plt.plot_lines([ave_acc_bir_no_rtf, ave_acc_bir_no_con, ave_acc_bir_no_int, ave_acc_bir_no_dis,
                                    ave_acc_gr_plus_gat],
                                   x_axes=dg_prop_list_onlybir, ylabel=ylabel,
                                   line_names=["BI-R - rtf", "BI-R - con", "BI-R - int", "BI-R - dis", "GR + gat"],
                                   colors=["maroon", "red", "darkgoldenrod", "green", "darkorange"], title=title,
                                   x_log=False, xlabel="Context gates: % of nodes gated", with_dots=True,
                                   ylim=(miny - marginy, maxy + marginy), h_lines=[BASE], h_labels=["None"],
                                   h_colors=["grey"])
        figure_list.append(figure)


    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))
