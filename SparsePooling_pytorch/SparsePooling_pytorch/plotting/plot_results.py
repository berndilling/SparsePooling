import matplotlib.pyplot as plt
import numpy as np
import torch 

linewidth = 2.0
fontsize = 20

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

def plot_sims(path, sims, labels, filename, plus_one = False):
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

        plt.plot(module_list, np.array(accs), linewidth = linewidth, label = labels[i])

plot_sims(path, sims, labels, filename)
plot_sims(path_CLAPP, sims_CLAPP, labels_CLAPP, filename_CLAPP, plus_one = True)


plt.scatter([0.75], 18.0, s=[50*linewidth], marker=(5, 1) , label = 'direct class',  c = ['black'])
# plt.scatter([6], 74.5, s=[50*linewidth], marker=(5, 1) , label = 'CLAPP',  c = ['grey'])

plt.legend()
plt.xlabel('classify from layer #')
plt.ylabel('STL-10 test accuracy [%]')

plt.savefig(savepath+'STL_class.pdf')

plt.show()


