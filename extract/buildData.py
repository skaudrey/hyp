#!/usr/bin/env python
# encoding: utf-8
'''
@author: MiaFeng
@contact: skaudrey@163.com
@file: buildData.py
@time: 2018/9/21 22:42
@desc: build test data and training data
'''
import pandas as pd
from Config import *
import numpy as np
# import geopandas as gpd
# from shapely.geometry import Point
# from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from util.fileUtil import CIoUtil


def makeScaleParamDf(chName, df):
    d = {}
    d['name'] = chName
    d['min'] = df[chName].min()
    d['max'] = df[chName].max()
    d['std'] = df[chName].std()
    dict_df = pd.DataFrame(d)
    return dict_df

def saveScaleParamDict(d, saveFileName):
    pass

class CFile2Img(object):
    def __init__(self,geo_res = 0.1):
        self.__geo_res = geo_res
        self.__ioTool = CIoUtil()

    def __makeMeshGrid(self,min_lat,max_lat,min_lon,max_lon):
        lon_size = int(( max_lon - min_lon )/self.__geo_res + 1 )
        lat_size = int(( max_lat - min_lat ) / self.__geo_res + 1 )
        lon_unit,lat_unit = \
            np.arange( min_lon, max_lon + self.__geo_res, self.__geo_res),\
            np.arange( min_lat, max_lat + self.__geo_res, self.__geo_res)
        lon_dict = dict( zip( np.around(lon_unit,1), np.arange(0, lon_size) ) )
        lat_dict = dict( zip( np.around(lat_unit,1), np.arange(0, lat_size) ) )
        return lat_dict,lon_dict

    def __cols2Index(self,colsName):
        '''
        mapping channels as integer indices. The indices used in channel figures are as same as these.
        :param colsName:
        :return:
        '''
        colsDict = pd.DataFrame({'colname':colsName,'idx':np.arange(0,len(colsName))})
        self.__ioTool.saveCSV(colsDict,'df2Img_colsMap_idx',category='data')
        return colsDict

    def __keepGeoNDecimal(self,df):
        df[['lat','lon']] = df[['lat','lon']].applymap(lambda x: round(x,1))
        df_geo = df[['lat','lon']]
        assert (len(df_geo) == len(df_geo.drop_duplicates()))
        return df

    def __filterDf(self,df,min_lat,max_lat,min_lon,max_lon):
        '''
        pick data in geometrical rectangle defined by minimum and maximum latitude and longitude
        :param df:
        :return:
        '''
        # print(df['lat'].dtypes)
        cond = (df['lat'] >= min_lat) & \
               (df['lat'] <= max_lat) & \
               (df['lon'] >= min_lon) & \
               (df['lon'] <= max_lon)
        df_reg = df[cond]
        return df_reg

    def smallDf2Grid(self,df_list,dataCols,reg_bouds):
        img_dict = {}
        self.__cols2Index(dataCols)

        for df_idx, (df, band_idx) in enumerate(zip(df_list,DATA_PREFIX)):
            df = self.__keepGeoNDecimal(df)
            regions_band = reg_bouds[reg_bouds['band_idx']==band_idx]
            regions_band.reset_index(drop=True, inplace=True)

            grid_regs = np.zeros((len(regions_band), LAT_SIZE, LON_SIZE, len(dataCols)),
                                 dtype=np.float16)

            for row_regions_band_idx, row_regions_band in regions_band.iterrows(): #遍历每个扫描带的每一个小区域，5'*5'
                df_reg = self.__filterDf( df,
                                          row_regions_band['min_lat'],
                                          row_regions_band['max_lat'],
                                          row_regions_band['min_lon'],
                                          row_regions_band['max_lon'])
                lat_dict,lon_dict = self.__makeMeshGrid(
                    row_regions_band['min_lat'],
                    row_regions_band['max_lat'],
                    row_regions_band['min_lon'],
                    row_regions_band['max_lon']
                )
                for chan_idx, chan_name in enumerate(dataCols):  # 遍历每个小区域的616个通道
                    for row_idx, row in df_reg.iterrows():
                        lat_tmp, lon_tmp = row['lat'], row['lon']
                        lat_idx, lon_idx = lat_dict[lat_tmp], lon_dict[lon_tmp]
                        tmp = row[chan_idx]
                        grid_regs[row_regions_band_idx][lat_idx][lon_idx][chan_idx] = np.float16(tmp)

            self.__ioTool.saveArray(grid_regs, DATA_PREFIX[df_idx])
        return img_dict

    def df2Grid(self,df_list,dataCols):
        img_dict = {}
        for df_idx,df in enumerate(df_list):
            df = self.__filterDf(self.__keepGeoNDecimal(df))
            grid_tmp = np.zeros((LAT_SIZE,LON_SIZE,len(dataCols)),dtype=np.float16)
            self.__cols2Index(dataCols)
            for idx,col in enumerate(dataCols): # 对有数据的部分进行填充
                for row_idx,row in df.iterrows():
                    lat_tmp,lon_tmp = row['lat'],row['lon']
                    lat_idx,lon_idx = self.__lat_dict[lat_tmp],self.__lon_dict[lon_tmp]
                    tmp = row[col]
                    grid_tmp[lat_idx][lon_idx][idx] = np.float16(tmp)
        self.__ioTool.saveArray(grid_tmp,DATA_PREFIX[df_idx])
        return img_dict


class CGeoUtil(object):
    '''
    Geometrical operations with GeoDataFrame, latitude and longitude are required.
    '''
    def __init__(self):
        self.__ioTool = CIoUtil()
    # def __getInterRegions(self,df_list):
    #     tmp = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(df_list[0])})
    #     for idx,geoItm in enumerate(df_list):
    #         if idx==0:
    #             continue
    #         tmp = gpd.overlay(tmp,gpd.GeoDataFrame({'geometry': gpd.GeoSeries(geoItm)}),how='intersection')
    #         # print(len(tmp.values[0][0]))
    #
    # def __getUnionRegions(self,df_list):
    #     tmp = gpd.GeoSeries(df_list[0])
    #     tmp = gpd.GeoDataFrame({'geometry': tmp})
    #
    #     for geoItm in df_list:
    #         tmp = gpd.overlay(tmp, gpd.GeoDataFrame({'geometry': gpd.GeoSeries(geoItm)}), how='union')
    #         print(len(tmp.values[0][0]))

        # return tmp.envelope

    def loadGridRegions(self):
        filename = '/df2fig/figReg.csv'

        df = pd.read_csv('%s/%s'%(getPath('data'),filename),sep=',',encoding='utf-8')

        df['band_idx'] = df['fig_name'].apply(lambda x:x[4:])

        return df




class CLoadData(object):
    def __init__(self):
        self.__ioTools = CIoUtil()

    # def loadDataAsPolygon(self,fileName):
    #     df = pd.read_csv("%s/%s" % (getPath(), fileName), sep=',')
    #     # print(list(zip(df.lon, df.lat))[:5])
    #     df_polygon = Polygon(list(zip(df.lon, df.lat)))
    #     return df_polygon

    def loadDataAsImg(self,df):
        scaleParam_df = pd.DataFrame({})
        for itm in df.columns.tolist():
            tmpParam_df = makeScaleParamDf(itm, df)
            scaleParam_df = pd.concat([scaleParam_df, tmpParam_df], ignore_index=True)

    # def loadGeoData(self,fileName, category='raw'):
    #     df = pd.read_csv("%s/%s" % (getPath(), fileName), sep=',')
    #     if category == 'raw':
    #         return df
    #     df['Coordinates'] = list(zip(df.lon, df.lat))
    #     df['Coordinates'] = df['Coordinates'].apply(Point)
    #     gdf = gpd.GeoDataFrame(df, geometry='Coordinates')
    #     return gdf

    def getEnvolope(self,fileName):
        '''
        load the boundaries saved in .m file.
        :param fileName:
        :return:
        '''
        boudEnv = self.__ioTools.loadModel(fileName)
        return boudEnv.total_bounds

    def manage2Area(self,df):
        '''
        the window data only contain clouds
        :param df:
        :return:
        '''
        df.sort_values(by=['lat','lon'],ascending=[False,True],inplace=True)
        df.reset_index(drop=True,inplace=True)
        # group_count = []
        # df['lon_int'] = df['lon'].map(lambda x:int(np.floor(x)))
        # df['lat_int'] = df['lat'].map(lambda x: int(np.floor(x)))
        # print('the length of longitude --> %d' % len(df['lon_int'].unique()))
        # print('the length of latitude --> %d' % len(df['lat_int'].unique()))
        # df_lat_groups = df[['lat','lat_int', 'lon']].groupby('lat_int')
        # for name,group in df_lat_groups:
        #     group_count.append(len(group))
        # print(np.mean(group_count))
        return df

    def sortByGeo(self,df):
        '''
        sort data with latitude descend, longitude ascend
        :param df:
        :return:
        '''
        df.sort_values(by=['lat','lon'],ascending=[False,True],inplace=True)
        df.reset_index(drop=True,inplace=True)
        return df

    def getSeaData(self,df_label):
        '''
        split data by topologies, and return data in sea topologies.
        :param df_label:
        :return:
        '''
        df_sea, df_sea_has_cloud, df_sea_no_cloud = [], [], []
        label_list = df_label[df_label['regidx']].values.tolist()
        for idx in label_list:
            df_tmp = df_label(df_label['regidx'] == idx)
            df_sea.append(df_tmp[df_tmp['topo'] == 0])
            df_sea_no_cloud.append(df_tmp[df_tmp['cloud'] == 0])
            df_sea_has_cloud.append(df_tmp[df_tmp['cloud'] == 1])
        return df_sea, df_sea_has_cloud, df_sea_no_cloud

    def loadDataOfAllRegion(self,iYear, iMonth, category='geo',isSort=False):
        '''
        load all data regions, the filename of them are named by year, month, date and belt indices.
        :param iYear: integer
        :param iMonth: integer
        :return:
        '''
        gdf_list = []
        date_list = [7, 8, 9]
        belt_list = [1, 2]
        # iterate all data regions
        for date_idx in date_list:
            for belt_idx in belt_list:
                fileName = '%d%02d%02d_%d.csv' % (iYear, iMonth, date_idx, belt_idx)
                gdf_tmp = None
                try:
                    if category == 'geo':
                        gdf_tmp = self.loadGeoData(fileName, category == 'geo')
                    elif category == 'polygon':
                        gdf_tmp = self.loadDataAsPolygon(fileName)
                    elif category == 'raw':
                        gdf_tmp = self.loadGeoData(fileName)
                finally:
                    if isSort:  # sort as latitude descend and longitude ascend
                        gdf_tmp = self.manage2Area(gdf_tmp)
                gdf_list.append(gdf_tmp)
        return gdf_list


def getRegCentroids(df):
    df_clust = df[df.columns[:-5].tolist()].values
    df_clust = StandardScaler().fit_transform(df_clust)
    db = DBSCAN(eps=0.001, min_samples=8,metric='cosine',algorithm='brute').fit(df_clust)
    print("The centroids of classification --> ")
    print(np.unique(db.labels_))
    df['regidx'] = db.labels_
    each_label_size = []
    for idx in np.unique(db.labels_):
        each_label_size.append(len(df[df['regidx']==idx]))

    return df


if __name__=='__main__':
    from extract.fileProcess import CRaw2CSV
    # getEnvolope('inter_boud')
    # df = loadGeoData('20150808_2.csv')
    loader = CLoadData()
    gdf_all = loader.loadDataOfAllRegion(2015,8,category='raw',isSort=False)
    geoTooler = CGeoUtil()
    reg_bouds = geoTooler.loadGridRegions()
    # from extract.visual import showDataOnMap
    # showDataOnMap(gdf_all,cols='ch921')
    imager = CFile2Img()
    rawer = CRaw2CSV()
    imager.smallDf2Grid(gdf_all,rawer.loadChanIndex(),reg_bouds)

    # geoTooler = CGeoUtil()
    # boundary = geoTooler.getInterRegions(gdf_all[1:])
    # saveModel(boundary, 'inter_boud')
    #
    # print(boundary)
    # saveModel(boundary,'union_boud')
    # gdf_inter = getInterRegions(gdf_all)
    # df_1_labeled = getRegCentroids(gdf_all[0])
    # df_sea, df_sea_has_cloud, df_sea_no_cloud = loader.getSeaData(df_1_labeled)
    # # getRegCentroids()