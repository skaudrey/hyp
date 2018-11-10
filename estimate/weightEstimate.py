# -*- coding: utf-8 -*-
# @File     : weightEstimate.py
# @Software : PyCharm   
__time__ = '10/29/184:46 PM'
__author = 'MiaFeng'

from Config import *
# from util.fileUtil import CIoUtil
# from sklearn.externals import joblib
import joblib
from util.fileUtil import makeFileDescrip
from util.fig_util import plotWeights
import numpy as np

def loadWeights(modelName,learning_rate,epoch,batch_size):
    '''

    :param modelName: such as 'chan2'
    :param batch_size:
    :return:
    '''
    fileDescrip = makeFileDescrip(modelName,learning_rate,epoch,batch_size)

    weights = joblib.load('%s/%s'%(getPath(modelName),fileDescrip))

    print(len(weights))

    return weights

if __name__=='__main__':
    modelName, learning_rate, epoch, batch_size = 'chan3',0.01,150,50

    weights = loadWeights(modelName, learning_rate, epoch, batch_size)

    layer_idx = 16

    weights_plot = weights[15][0]

    kernel_num = np.shape(weights_plot)[-1]

    max_weights = np.max(weights_plot)
    min_weights = np.min(weights_plot)

    for kernel_idx in range(kernel_num):
        savefigName = '%s_layer_%d_k_%d_w'%(modelName,layer_idx,kernel_idx+1)

        plotWeights(weights_plot[:,:,:,kernel_idx],savefigName=savefigName)

    pass