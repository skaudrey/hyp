#!/usr/bin/env python
# encoding: utf-8
'''
@author: MiaFeng
@contact: skaudrey@163.com
@file: compModel.py
@time: 2018/9/25 10:24
@desc:
'''

from keras.layers import Conv1D,Conv2D,Conv2DTranspose,MaxPooling1D,MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
# from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras.losses import MSE
from keras.metrics import MSE
from keras.layers import Dropout,Reshape,Permute,SpatialDropout1D,SpatialDropout2D,UpSampling1D,UpSampling2D,BatchNormalization
from util.fileUtil import getPath
from util.fileUtil import CIoUtil
import numpy as np
from extract.buildData import CGeoUtil
from keras.models import save_model,load_model
from sklearn.metrics import mean_squared_error
import joblib

def add_conv1d_trans(model,kernel_size_upsamp,kerne_size_conv1d=None,interpolation = 'nearest'):
    '''
    Supporting either upsampling by bilinear of nearest method or conv1D after expandding images by upsamling2D
    :param model:
    :param kernel_size_upsamp:
    :param kerne_size_conv1d:
    :param interpolation:
    :return:
    '''
    model.add(UpSampling2D(size=kernel_size_upsamp, interpolation=interpolation))
    if kerne_size_conv1d!=None:
        model.add(Reshape(-1,IMG_LON_SIZE*IMG_LAT_SIZE))
        model.add(Permute(2,1))
        model.add(Conv1D(IMG_LAT_SIZE*IMG_LON_SIZE,kernel_size=kerne_size_conv1d))
        model.add(Permute(2, 1))
        model.add(Reshape(-1, IMG_LON_SIZE * IMG_LAT_SIZE,1))
    return model


def compModel_2(saveModelFigName = ''):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(2, 2), strides=(2, 2),
                     activation='relu', name='comp_space_1',input_shape=INPUT_SHAPE))

    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(16, 16), strides=(1, 1),
                     activation='relu', name='comp_space_2'))
    model.add(SpatialDropout2D(DROPOUT_RATE))

    model.add(BatchNormalization())
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', name='comp_space_3'))

    model.add(BatchNormalization())
    model.add(Conv2D(8, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', name='comp_space_4'))

    model.add(BatchNormalization())
    model.add(Conv2D(8, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', name='comp_space_5'))
    model.add(BatchNormalization())
    model.add(Conv2D(8, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', name='comp_space_6'))
    model.add(BatchNormalization())

    print('the output shape after spatial compression --> ')
    print(model.output_shape[1:])

    # =======================   reconstruction    ========================
    # ----------------  spatial reconstruction  ----------------
    # model.add(Conv2DTranspose(8, kernel_size=(128, 128), strides=(1, 1), activation='relu', name='res_space_4_1'))
    # model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
    #                  activation='relu', name='res_space_4_2'))
    # model.add(SpatialDropout2D(DROPOUT_RATE))
    # model.add(BatchNormalization())


    model.add(Conv2DTranspose(32, kernel_size=(32,32), strides=(1, 1), activation='relu', name='res_space_3_1'))
    model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', name='res_space_3_2'))
    model.add(SpatialDropout2D(DROPOUT_RATE))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(64, kernel_size=(16, 16), strides=(1, 1), activation='relu', name='res_space_2_1'))
    model.add(Conv2D(256, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', name='res_space_2_2'))
    model.add(SpatialDropout2D(DROPOUT_RATE))
    model.add(BatchNormalization())

    model.add(
        Conv2DTranspose(256, kernel_size=(4, 4), strides=(1, 1), activation='relu', name='res_space_1_1'))
    model.add(Conv2D(616, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', name='res_space_1_2'))
    model.add(SpatialDropout2D(DROPOUT_RATE))
    # model.add(Conv2D(616, kernel_size=(2, 2), strides=(1, 1),
    #                  activation='relu', name='res_space_1_3'))
    # model.add(Conv2D(616, kernel_size=(2, 2), strides=(1, 1),
    #                  activation='relu', name='res_space_1_4'))

    print('the output shape after spatial reconstruction --> ')
    print(model.output_shape[1:])
    assert model.output_shape[1:3] == (IMG_LAT_SIZE, IMG_LON_SIZE)

    return model

def compModel_3(saveModelFigName = ''):
    model = Sequential()

    # =======================   compression    ========================

    model.add(Conv2D(3, kernel_size=(8, 8), strides=(2, 2),
                     activation='relu', name='comp_space_1',input_shape=INPUT_SHAPE))

    model.add(BatchNormalization())
    model.add(Conv2D(2, kernel_size=(4, 4), strides=(1, 1),
                     activation='relu', name='comp_space_2'))

    model.add(BatchNormalization())
    model.add(Conv2D(2, kernel_size=(4, 4), strides=(1, 1),
                     activation='relu', name='comp_space_3'))

    model.add(BatchNormalization())
    model.add(Conv2D(2, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', name='comp_space_4'))

    print('the output shape after spatial compression --> ' )
    print(model.output_shape[1:])

    # =======================   reconstruction    ========================
    # ----------------  spatial reconstruction  ----------------
    model.add(Conv2DTranspose(2, kernel_size=(16,16), strides=(1, 1), activation='relu',name='res_space_6'))

    model.add(
        Conv2DTranspose(4, kernel_size=(7, 7), strides=(1, 1), activation='relu', name='res_space_5'))

    model.add(
        Conv2DTranspose(4, kernel_size=(7, 7), strides=(1, 1), activation='relu', name='res_space_4'))

    model.add(
        Conv2DTranspose(8, kernel_size=(5,5), strides=(1, 1), activation='relu', name='res_space_3'))

    model.add(
        Conv2DTranspose(8, kernel_size=(5, 5), strides=(1, 1), activation='relu', name='res_space_2'))

    model.add(
        Conv2DTranspose(616, kernel_size=(2, 2), strides=(1, 1), activation='relu', name='res_space_1'))

    print('the output shape after spatial reconstruction --> ' )
    print(model.output_shape[1:])
    assert model.output_shape[1:3] == (IMG_LAT_SIZE, IMG_LON_SIZE)


    return model


def compModel_1(chan_num, saveModelFigName = ''):
    model = Sequential()

    # =======================   compression    ========================
    # ----------------   adjust shape for spectral convolution ----------------
    # model.add(Reshape((-1, CHAN_DIM), input_shape=INPUT_SHAPE)) # change tensor's shape in order to do conv1d
    # # output shape: (None, IMG_LAT_SIZE*IMG_LON_SIZE,CHAN_DIM)
    # model.add(Permute((2, 1))) # do permutation in order to support just convolution on channels
    # # ----------------  spectral convolution of compression  ----------------
    # model.add(Conv1D(IMG_LAT_SIZE*IMG_LON_SIZE,kernel_size=5,strides=3,activation='relu',name='comp_spect_1'))
    # model.add()
    # model.add(SpatialDropout1D(DROPOUT_RATE))

    # model.add(BatchNormalization())
    # model.add(Conv1D(IMG_LAT_SIZE*IMG_LON_SIZE, kernel_size=5, strides=2, activation='relu',name='comp_spect_2'))
    # # model.add(SpatialDropout1D(DROPOUT_RATE))
    #
    # model.add(BatchNormalization())
    # model.add(Conv1D(IMG_LAT_SIZE*IMG_LON_SIZE, kernel_size=3, strides=3, activation='relu',name='comp_spect_3'))
    # # model.add(SpatialDropout1D(DROPOUT_RATE))
    #
    # model.add(BatchNormalization())
    # model.add(Conv1D(IMG_LAT_SIZE*IMG_LON_SIZE, kernel_size=3, strides=2, activation='relu',name='comp_spect_4'))
    # # model.add(SpatialDropout1D(DROPOUT_RATE))
    #
    # model.add(BatchNormalization())
    # model.add(Conv1D(IMG_LAT_SIZE * IMG_LON_SIZE, kernel_size=2, strides=1, activation='relu', name='comp_spect_5'))
    # model.add(Conv1D(IMG_LAT_SIZE * IMG_LON_SIZE, kernel_size=2, strides=1, activation='relu', name='comp_spect_6'))
    #
    # print('the output shape after spectrum compression before reshape --> ' )
    # print(model.output_shape[1:])
    # # ----------------  restore data shape for spatial convolution  ----------------
    # model.add(Permute((2, 1)))
    # comp_spect = model.add(Reshape((IMG_LAT_SIZE,IMG_LON_SIZE, -1)))
    #
    # # ----------------  spatial convolution  ----------------
    # print('the output shape after spectral compression --> ' )
    # print(model.output_shape[1:])
    # spect_comp_size = model.output_shape[-1]
    # assert model.output_shape[1:3]==(IMG_LAT_SIZE,IMG_LON_SIZE)

    model.add(Conv2D(chan_num+2, kernel_size=(15, 15), strides=(1, 1),
                     activation='relu', name='comp_space_1',input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(1,1)))
    model.add(BatchNormalization())

    model.add(Conv2D(chan_num+1, kernel_size=(7, 7), strides=(1, 1),
                     activation='relu', name='comp_space_2'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(1,1)))
    model.add(BatchNormalization())


    model.add(Conv2D(chan_num, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', name='comp_space_3'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    model.add(BatchNormalization())

    model.add(Conv2D(chan_num, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', name='comp_space_4'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    model.add(BatchNormalization())
    print('the output shape after spatial compression --> ' )
    print(model.output_shape[1:])

    # =======================   reconstruction    ========================
    # ----------------  spatial reconstruction  ----------------
    model.add(Conv2DTranspose(4, kernel_size=(15,15), strides=(1, 1), activation='relu',name='res_space_4'))
    # model.add(SpatialDropout2D(DROPOUT_RATE))
    # model.add(BatchNormalization())

    model.add(
        Conv2DTranspose(4, kernel_size=(9, 9), strides=(1, 1), activation='relu', name='res_space_3'))
    # model.add(SpatialDropout2D(DROPOUT_RATE))
    # model.add(BatchNormalization())

    model.add(
        Conv2DTranspose(8, kernel_size=(6, 6), strides=(1, 1), activation='relu', name='res_space_2'))

    model.add(
        Conv2DTranspose(616, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='res_space_1'))

    print('the output shape after spatial reconstruction --> ' )
    print(model.output_shape[1:])
    assert model.output_shape[1:3] == (IMG_LAT_SIZE, IMG_LON_SIZE)

    # ----------------  spectrum reconstruction  ----------------
    # ----------------   adjust shape for spectral reconstruction ----------------
    # model.add(Reshape((-1, spect_comp_size,1)))  # add one dimension for using conv2DTranspose or Upsampling2D
    # model.add(Permute((2, 1,3)))  # do permutation in order to support just convolution on channels
    #
    # model = add_conv1d_trans(model,interpolation='bilinear',kernel_size_upsamp=(11,1))
    # # model.add(SpatialDropout2D(DROPOUT_RATE))
    #
    # model = add_conv1d_trans(model, interpolation='bilinear', kernel_size_upsamp=(2, 1))
    # # model.add(SpatialDropout2D(DROPOUT_RATE))
    #
    # model = add_conv1d_trans(model, interpolation='bilinear', kernel_size_upsamp=(2, 1))
    # # output shape: (None, IMG_LAT_SIZE*IMG_LON_SIZE,CHAN_DIM)
    #
    #
    # # ----------------  restore data shape for spectrum reconstruction  ----------------
    # model.add(Reshape((CHAN_DIM,IMG_LAT_SIZE*IMG_LON_SIZE)))
    # model.add(Permute((2, 1)))
    # res_spect = model.add(Reshape((IMG_LAT_SIZE, IMG_LON_SIZE, -1),name='output'))
    # # ----------------  spatial convolution  ----------------
    # print('the output shape after reconstruction --> ')
    # print(model.output_shape[1:])
    # assert model.output_shape[1:] == (IMG_LAT_SIZE, IMG_LON_SIZE,CHAN_DIM)
    #


    return model

def loadData(trainBandIdx,testBandIdx):
    geoTooler = CGeoUtil()
    reg_bouds = geoTooler.loadGridRegions()

    train_file_name = '%s/%s_shape_(%d, 51, 51, 616).bin' % (
    getPath('figData'), trainBandIdx, len(reg_bouds[reg_bouds['band_idx'] == trainBandIdx]))
    test_file_name = '%s/%s_shape_(%d, 51, 51, 616).bin' % (
    getPath('figData'), testBandIdx, len(reg_bouds[reg_bouds['band_idx'] == testBandIdx]))

    train_array = ioTooler.loadArray(train_file_name)
    test_array = ioTooler.loadArray(test_file_name)
    return train_array, test_array


def train_model(model,X_train,X_test,test_date,learning_rate,epoch,batch_size,isTestCorrect=False,isSaveModel=False):

    if isTestCorrect:
        X_train,X_test = X_train[:10,:,:,:],X_test[:10,:,:,:]

    # split train and test data
    Y_train, Y_test = X_train, X_test

    # if want to use SGD, first define sgd, then set optimizer=sgd
    sgd = SGD(lr=learning_rate, decay=learning_rate/epoch, momentum=0, nesterov=True)

    # select loss\optimizer\
    model.compile(loss=MSE,
                  optimizer=sgd, metrics=['MSE'])

    # input data to model and train
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                        validation_split=0.2, verbose=0, shuffle=True)

    # evaluate the model
    loss, mse = model.evaluate(X_test, Y_test, verbose=0)

    y_pred = model.predict(X_test,batch_size=batch_size)

    print('test date <%s> with lr <%f>, epoch <%d>, batch_size <%d>:' % (test_date,learning_rate,epoch,batch_size))

    print('Test loss: (lr-%f, epoch-%d, batch_size-%d) -->%f'%(learning_rate,epoch,batch_size,loss))
    print('Test loss: (lr-%f, epoch-%d, batch_size-%d) -->%f'%(learning_rate,epoch,batch_size,loss))

    model_descrip = makeModelDescrip(test_date,learning_rate,epoch,batch_size)

    if isSaveModel:
        modelName = '%s/model_%s' % (getPath('model'), model_descrip)
        save_model(model, filepath=modelName)
        print('save model --> %s'% modelName)


    return model,history,loss,mse,y_pred


def estimateAcc(model,X_test):
    X_res = model.predict(X_test)
    return mean_squared_error(X_test.reshape((-1,CHAN_DIM)),X_res.reshape((-1,CHAN_DIM)))


def makeModelDescrip(time,learning_rate,epoch,batch_size):
    return '%s_lr%s_epc%d_batch%d'%(time,learning_rate,epoch,batch_size)

def saveWeights(model,fileDescrip,test_date):
    weights_list = []

    for layer in model.layers:
        weights = layer.get_weights()
        weights_list.append(weights)

    filename = '%s/weights_%s' % (getPath(test_date), fileDescrip)


    joblib.dump(weights_list, filename)
    print('save weights --> %s' % filename)

def saveResult(test_date,history,loss,mse,y_test,y_pred,fileDescrip):
    result = {
        'loss': loss,
        'mse': mse,
        'epoch': history.params['epochs'],
        'batch_size':history.params['batch_size'],
        'val_loss' : history.history['val_loss'],
        'val_mse': history.history['val_mean_squared_error'],
        'train_loss': history.history['loss'],
        'train_mse': history.history['mean_squared_error'],
        'y_test':y_test,
        'y_pred':y_pred
    }
    filename = '%s/result_%s.pkl' % (getPath(test_date), fileDescrip)
    joblib.dump(result, filename)
    print('save result --> %s'%filename)


def saveChanDiffRes(chan_diff,fileDescrip):

    filename = '%s/result_%s.pkl' % (getPath('result'), fileDescrip)
    joblib.dump(chan_diff, filename)
    print('save result --> %s' % filename)

def hypfineTuning(chan_num,learning_rate_list,epoch_list,batch_size_list,X_train,X_test,model,isTestCorrect=False):

    chan_diff_res = []
    chan_diff_label = []


    for learning_rate in learning_rate_list:
        for batch_size in batch_size_list:
            chan_diff_label_tmp = '(%s,%d)' %(chan_num[-1],batch_size)
            chan_diff_res_tmp = []
            for epoch in epoch_list:

                reg_tooler, history, loss, mse,y_pred =train_model(
                    model,
                    X_train,
                    X_test,
                    chan_num,
                    learning_rate,
                    epoch,
                    batch_size,
                    isTestCorrect=isTestCorrect,
                    isSaveModel=True)
                fileDescrip = makeModelDescrip(chan_num, learning_rate, epoch, batch_size)
                saveWeights(reg_tooler,fileDescrip,chan_num)
                saveResult(chan_num, history, loss, mse,X_test,y_pred,fileDescrip)
                chan_diff_res_tmp.append(mse)
                print(80*'-')
            chan_diff_res.append(chan_diff_res_tmp)
            chan_diff_label.append(chan_diff_label_tmp)

    chan_diff = {
        'chan_diff_mse':chan_diff_res,
        'chan_diff_label':chan_diff_label
    }
    saveChanDiffRes(chan_diff,'chan_diff_res')





def trainPerDate():
    learning_rate_list = [1e-2]
    epoch_list = [70,80]
    batch_size_list = [40,50]
    model = None
    modelName = 'model_1'

    for chan_num_tmp in range(3,10,1):

        if modelName == 'model_1':
            model = compModel_1(saveModelFigName='HCR-struct_%s'%modelName,chan_num=chan_num_tmp)
        elif modelName == 'model_2':
            model = compModel_2(saveModelFigName='HCR-struct_%s'%modelName)
        elif modelName == 'model_3':
            model = compModel_3(saveModelFigName='HCR-struct_%s'%modelName)

        test_date_list = ['0807', '0808', '0809']
        X_train, X_test = loadData('%s_1' % test_date_list[0],
                                   '%s_2' % test_date_list[0])

        X_test = X_test[:FIG_NUM, :, :, :]

        for TEST_DATE in test_date_list[1:]:
            # load data
            X_train_tmp, X_test_tmp = loadData('%s_1' % TEST_DATE,
                                               '%s_2' % TEST_DATE)

            X_test_tmp = X_test_tmp[:FIG_NUM, :, :, :]

            X_train = np.vstack((X_train,X_train_tmp))

            X_test = np.vstack((X_test,X_test_tmp))

            np.random.shuffle(X_train)

        hypfineTuning('chan%d'%chan_num_tmp, learning_rate_list, epoch_list, batch_size_list, X_train, X_test, model,
                      isTestCorrect=False)


if __name__=='__main__':
    IMG_LAT_SIZE, IMG_LON_SIZE = 51, 51
    FIG_NUM = 34
    CHAN_DIM = 616
    INPUT_SHAPE = (IMG_LAT_SIZE, IMG_LON_SIZE, CHAN_DIM)
    DROPOUT_RATE = 0.5
    ioTooler = CIoUtil()
    trainPerDate()





