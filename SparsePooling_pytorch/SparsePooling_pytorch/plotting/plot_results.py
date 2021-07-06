import matplotlib.pyplot as plt
import numpy as np
import torch 

linewidth = 2.0
fontsize = 20

def plot_sims(path, modules, module_list, sims, labels, filename, colors = None, linestyles = None, plus_one = False):
        for i, sim in enumerate(sims):
            accs = []
            accstop5 = []
            for m in modules:
                if plus_one:
                    mn = m+1
                else:
                    mn = m
                acc = np.load(path+sims[i]+filename+str(mn)+'.npy')
                accs.append(acc[0])
                accstop5.append(acc[1])
            if colors == None and linestyles == None:
                plt.plot(module_list, np.array(accs), linewidth = linewidth, label = labels[i])
            else:
                plt.plot(module_list, np.array(accs), color = colors[i], linestyle = linestyles[i], linewidth = linewidth, label = labels[i])

def plot_results_3_layer():
    savepath = '/Users/Bernd/Documents/PhD/Presentations/LabMeeting_March21/'

    path = '/Users/Bernd/Documents/PhD/Projects/SparsePooling/SparsePooling_pytorch/logs/'
    sims = ['SparsePooling_MaxPool_through_SFA_end_to_end_supervised_2/', 
            'SparsePooling_stack_SC_SFA/', 
            'SparsePooling_stack_SC_SFA_random_init/',
            'SparsePooling_stack_SC_SFA_stl10/']
    labels = ['Supervised BP', 'SC-SFA', 'SC-SFA STL-10', 'Random init']
    filename = 'classification_results_values_layer_'

    path_CLAPP = '/Users/Bernd/Documents/PhD/Projects/gim-local/logs/'
    sims_CLAPP = ['vgg_6_layers_6_modules_hinge_2/']
    labels_CLAPP = ['CLAPP']
    filename_CLAPP = 'classification_results_values_'

    fig = plt.figure()
    ax = plt.gca()

    modules = range(6)
    module_list = np.array([*modules])+1

    plot_sims(path, modules, module_list, sims, labels, filename)
    plot_sims(path_CLAPP, modules, module_list, sims_CLAPP, labels_CLAPP, filename_CLAPP, plus_one = True)

    plt.scatter([0.75], 18.0, s=[50*linewidth], marker=(5, 1) , label = 'direct class',  c = ['black'])
    # plt.scatter([6], 74.5, s=[50*linewidth], marker=(5, 1) , label = 'CLAPP',  c = ['grey'])

    plt.legend()
    plt.xlabel('classify from layer #')
    plt.ylabel('STL-10 test accuracy [%]')

    plt.savefig(savepath+'STL_class.pdf')

    plt.show()

def plot_results_vgg6(plot_CLAPP=True):
    savepath = '/Users/Bernd/Documents/PhD/Presentations/BMI_progress_report_21/'

    path = '/Users/Bernd/Documents/PhD/Projects/SparsePooling/SparsePooling_pytorch/logs/'
    sims = ['SparsePooling_stack_SC_MaxPool_kWTA_vgg6_end_to_end_supervised/',
            'SparsePooling_stack_SC_MaxPool_kWTA_vgg6_random_init/',
            'SparsePooling_stack_SC_MaxPool_kWTA_vgg6/',
            'SparsePooling_stack_SC_SFA_kWTA_vgg6/',
            ]
    labels = ['Supervised BP','Random init', 'SC-MaxPool', 'SC-SFA']
    colors = ['red', 'gray', 'purple', 'purple']
    linestyles = ['-', ':', '-', '--']
    filename = 'classification_results_values_layer_'

    path_CLAPP = '/Users/Bernd/Documents/PhD/Projects/gim-local/logs/'
    sims_CLAPP = ['vgg_6_layers_6_modules_hinge_2/']
    labels_CLAPP = ['CLAPP']
    filename_CLAPP = 'classification_results_values_'

    fig = plt.figure()
    ax = plt.gca()

    modules = [0, 2, 3, 5, 7, 9]
    module_list = np.array([*modules])+1
    
    modules_CLAPP = range(6) # cause CLAPP does not count the pooling/SFA layers
    module_list_CLAPP = np.array([*modules_CLAPP])+1

    plt.scatter([0.75], 18.0, s=[50*linewidth], marker=(5, 1) , label = 'Direct class.',  c = ['black'])
    # plt.scatter([6], 74.5, s=[50*linewidth], marker=(5, 1) , label = 'CLAPP',  c = ['grey'])

    plt.plot([1, 6],[10., 10.], color = 'gray', linestyle = '-.', label = 'Chance')

    plot_sims(path, modules, module_list_CLAPP, sims, labels, filename, colors = colors, linestyles = linestyles)
    if plot_CLAPP:
        plot_sims(path_CLAPP, modules_CLAPP, module_list_CLAPP, sims_CLAPP, labels_CLAPP, filename_CLAPP, colors = ['blue'], linestyles = ['-'], plus_one = True)

    plt.ylim([7, 73])
    plt.legend()
    plt.xlabel('classify from layer #')
    plt.ylabel('STL-10 test accuracy [%]')

    plt.savefig(savepath+'STL_class.pdf')

    plt.show()

##

#plot_results_3_layer()
plot_results_vgg6(plot_CLAPP=False)

