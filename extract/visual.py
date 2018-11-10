#!/usr/bin/env python
# encoding: utf-8
'''
@author: MiaFeng
@contact: skaudrey@163.com
@file: visual.py
@time: 2018/9/25 11:20
@desc:
'''
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd

def showDataOnMap(gdf_list,cols):
    '''
    plot radiance in specific channel indicated by "cols"
    :param gdf:
    :param cols: specify the shown channel's name
    :param cmap:
    :return:
    '''
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    print(world.continent)
    cmap_list = ['OrRd','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r']
    # We restrict to Asia and Oceania
    ax = world[(world.continent == 'Asia').tolist() or (world.continent == 'Oceania').tolist()].\
        plot(color='white', edgecolor='black')
    for idx,gdf in enumerate(gdf_list):
        gdf.plot(ax=ax,column= cols, cmap=cmap_list[idx])

    plt.show()
def showDataTopo(gdf):
    '''
    plot topology
    :param gdf: GeoDataFrame
    :return:
    '''
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    topos = gdf.topo.unique().tolist()
    topos.sort()
    topos[0],topos[1],topos[2],topos[3],topos[4] = 'sea', 'land low','land high','land water low','land water high'
    colors = ["#0092c7","#f3e59a","#9fe0f6",
              "#f3b59a","#f29c9c","#22c3aa"]
    # We restrict to Asia and Oceania
    ax = world[(world.continent == 'Asia').tolist() or (world.continent == 'Oceania').tolist()]. \
        plot(color='white', edgecolor='black')
    for idx in range(len(topos)):
        gdf[gdf['topo']==idx].plot(ax=ax, column='topo', color=colors[idx],label=topos[idx])
    plt.legend(loc='best')
    plt.show()


if __name__=='__main__':
    import pandas as pd
    df = pd.DataFrame({})
    showDataTopo(df)
    showDataOnMap(df,'ch921')