#!/usr/bin/env python
# encoding: utf-8
'''
@author: MiaFeng
@contact: skaudrey@163.com
@file: pcaBaseline.py
@time: 2018/10/21 16:34
@desc:
'''

from util.fileUtil import CIoUtil
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from Config import getPath
from extract.buildData import CGeoUtil
ioTooler = CIoUtil()
FIG_NUM = 34
import joblib
def loadData(trainBandIdx,testBandIdx):

    geoTooler = CGeoUtil()
    reg_bouds = geoTooler.loadGridRegions()

    train_file_name = '%s/%s_shape_(%d, 51, 51, 616).bin'%(getPath('figData'),trainBandIdx,len(reg_bouds[reg_bouds['band_idx']==trainBandIdx]))
    test_file_name = '%s/%s_shape_(%d, 51, 51, 616).bin'%(getPath('figData'),testBandIdx,len(reg_bouds[reg_bouds['band_idx']==testBandIdx]))

    train_array = ioTooler.loadArray(train_file_name)
    test_array = ioTooler.loadArray(test_file_name)
    return train_array,test_array

def loadAllData():
    test_date_list = ['0807', '0808', '0809']
    X_train, X_test = loadData('%s_1' % test_date_list[0],
                               '%s_2' % test_date_list[0])

    X_test = X_test[:FIG_NUM, :, :, :]

    for TEST_DATE in test_date_list[1:]:
        # load data
        X_train_tmp, X_test_tmp = loadData('%s_1' % TEST_DATE,
                                           '%s_2' % TEST_DATE)

        X_test_tmp = X_test_tmp[:FIG_NUM, :, :, :]

        X_train = np.vstack((X_train, X_train_tmp))

        X_test = np.vstack((X_test, X_test_tmp))

        np.random.shuffle(X_train)

    return X_train,X_test

def trainPCA(X,chan_tar_num=None,geo_tar_num=None,savemodel=False):
    pca = PCA()
    modelDescrip = ''
    if chan_tar_num != None:
        X_train = reshapaData(X,'chan')
        modelDescrip = 'chan'
        print('compression spectrum data')
        pca = PCA(n_components=chan_tar_num)
        pca.fit(X_train)
        ioTooler.saveModel(pca,'hyp_pca_chan')
    elif geo_tar_num != None:
        '''
        compress the geophysical features after channel compression 
        '''
        modelDescrip = 'geo'
        print('compression spatial data')
        # X_train = reshapaData(X, 'geo')
        X_train = X
        pca = PCA(n_components=geo_tar_num)
        pca.fit(X_train)
        ioTooler.saveModel(pca, 'hyp_pca_geo')

    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)

    if savemodel:
        ioTooler.saveModel(pca, 'hyp_pca_%s'%modelDescrip)

    return pca.transform(X_train)


def comp(comp_type,X):
    modelName = 'hyp_pca_%s'%comp_type
    pca = ioTooler.loadModel(modelName)

    X_comp = pca.transform(X)

    return X_comp

def reconstruction(res_type,X_comp):
    modelName = 'hyp_pca_%s' % res_type
    pca = ioTooler.loadModel(modelName)

    X_res = pca.inverse_transform(X_comp)


    return X_res

def estimateAcc(X_res,X_raw):
    return mean_squared_error(X_raw,X_res)

def estimateCompRatio(X_res,X_raw):
    return X_res.size/X_raw.size*0.1/0.1



def reshapaData(X,reshapeType='chan'):
    '''
    reshape data to compress channels or geographical series
    :param X: (#regions, #lat_grid, #lon_grid, #channels)
    :param reshapeType:
    :return:
    '''
    reg_size,lat_size,lon_size,chan_size = X.shape[0],X.shape[1],X.shape[2],X.shape[3]
    if reshapeType=='chan':
        return X.reshape((reg_size*lat_size*lon_size,chan_size))
    elif reshapeType == 'geo':
        return (X.reshape((reg_size * lat_size * lon_size, chan_size))).T

def saveResult(chan_comp_num,geo_tar_num,geo_mse,chan_mse,mse,y_test,y_pred,fileDescrip):
    result = {
        'spect_comp_num': chan_comp_num,
        'geo_comp_num':geo_tar_num,
        'geo_mse':geo_mse,
        'spect_mse':chan_mse,
        'mse': mse,
        'y_test':y_test,
        'y_pred':y_pred
    }
    filename = '%s/result_%s.pkl' % (getPath('pca'), fileDescrip)
    joblib.dump(result, filename)
    print('save result --> %s'%filename)

if __name__ == '__main__':
    import numpy as np
    geo_tar_num = [22,22]

    X_train, X_test = loadAllData()

    isTestCorrect = True
    testSize = 90
    if isTestCorrect:
        X_train, X_test = X_train[:testSize, :, :, :], X_test[:testSize, :, :, :]

    # split train and test data
    Y_train, Y_test = X_train, X_test

    chan_comp_num = range(2,10,1)
    geo_mse = []
    spect_mse = []
    mse = []
    comp_ratio = 1
    y_pred = []

    for tar_num_tmp in chan_comp_num:

        pca_chan_train_comp = trainPCA(X_train,chan_tar_num=tar_num_tmp)
        trainPCA(pca_chan_train_comp.T,geo_tar_num=geo_tar_num[0]*geo_tar_num[1])


        compType = 'chan'
        pca_chan_comp = comp(compType,reshapaData(X_test,compType))
        pca_chan_res = reconstruction(compType,pca_chan_comp)
        chan_mse_tmp = estimateAcc(pca_chan_res,reshapaData(X_test,compType))
        print('MSE after reconstructing %s %.2f'%(compType, chan_mse_tmp))

        compType = 'geo'
        pca_geo_comp = comp(compType, pca_chan_comp.T)
        pca_geo_res = reconstruction(compType, pca_geo_comp)
        geo_mse_tmp = estimateAcc(pca_geo_res,pca_chan_comp.T)
        print('MSE after reconstructing %s %.2f'
              %(compType, geo_mse_tmp))

        pca_res = reconstruction('chan',pca_geo_res.T)

        mse_tmp = estimateAcc(pca_res,reshapaData(X_test,'geo').T)
        pca_res = pca_res.reshape((testSize, 51, 51, -1))
        print('MSE after reconstructing channels then geo --> %.2f' %(mse_tmp))
        print('compression ratio --> %.2f%%' % (
                estimateCompRatio(pca_geo_res, X_test)*100))
        comp_ratio = estimateCompRatio(pca_geo_res, X_test)
        print(80*'-')

        spect_mse.append(chan_mse_tmp)
        geo_mse.append(geo_mse_tmp)
        mse.append(mse_tmp)
        y_pred.append(pca_res)

    saveResult(chan_comp_num,geo_tar_num,geo_mse,spect_mse,mse,X_test,y_pred,'pca_chan_mse')




