# -*- coding: utf-8 -*-
# @File     : fig_util.py
# @Software : PyCharm   
__time__ = '10/29/185:33 PM'
__author = 'MiaFeng'

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Config import *
import matplotlib

rc = {
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 24,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    "grid.linewidth": 1.5,
    "lines.linewidth": 3.5,
    "patch.linewidth": .3,
    "lines.markersize": 7,
    "lines.markeredgewidth": 0,

    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": .5,
    "ytick.minor.width": .5,

    "xtick.major.pad": 7,
    "ytick.major.pad": 7

}
sns.set_context('talk',font_scale=1,rc = rc)

def plotWeights(matrixArray,grid_size = [28,22],savefigName=''):
    # fig = plt.figure()
    fig,ax = plt.subplots(grid_size[0],grid_size[1])
    plt.subplots_adjust(wspace = 1e-12,hspace=0.1)
    ax = ax.flatten()

    spect_num = np.shape(matrixArray)[-1]

    cmap = matplotlib.cm.jet

    norm = matplotlib.colors.Normalize(vmin=-0.034,vmax = 0.038)
    ax_tmp,idx_fig = None,0

    for idxRow in range(grid_size[0]):
        for idxCol in range(grid_size[1]):

            idx_fig = idxRow*grid_size[1] + idxCol

            mat_plot = matrixArray[:,:,idx_fig]

            ax_tmp = ax[idx_fig].imshow(mat_plot,cmap=cmap,norm=norm)
            ax[idx_fig].axis('off')
            ax[idx_fig].set_xticks([])
            ax[idx_fig].set_yticks([])

    position = fig.add_axes([0.15,0.07,0.7,0.03])
    fig.colorbar(ax_tmp,cax=position,orientation='horizontal')
    # plt.tight_layout(h_pad = 0.00001,w_pad = 0.00001)




    if len(savefigName)>0:
        plt.savefig('%s/%s.png'%(getPath('showfig'),savefigName),dpi = 600)

    plt.show()