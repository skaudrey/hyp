import plotly.plotly as py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from Config import *
import plotly

def setPlotlyAccount():
    plotly.tools.set_credentials_file(username=PLOTLY_USR_NAME, api_key=PLOTLY_API_KEY)

def getColorRGB(colorMapName,markerNum):

    values = range(markerNum)

    jet = cm = plt.get_cmap(colorMapName)
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    colorsList = []
    for i in range(markerNum):
        colorVal = scalarMap.to_rgba(values[i])

        colorsList.append( "rgb(%4.2f,%4.2f,%4.2f)"%(colorVal[0], colorVal[1], colorVal[2]))
    colorsList.append('lightgrey')
    return colorsList


def bubbleRankMap(comVertexList,df,plotFea):
    setPlotlyAccount()

    limits = comVertexList
    colors = getColorRGB('jet',len(comVertexList))
    # colors = ["rgb(0,116,217)", "rgb(255,65,54)", "rgb(133,20,75)", "rgb(255,133,27)", "lightgrey"]
    communities = []
    scale = 10

    for i,vertex in enumerate(limits):
        # print('the size of community --> %d'%len(vertex))
        indices = vertex
        try:
            df_sub = df.iloc[indices]
        except:
            print(indices)
        community = dict(
            type='scattergeo',
            locationmode='',
            lon=df_sub['lon'],
            lat=df_sub['lat'],
            text='',
            marker=dict(
                size=df_sub[plotFea] * scale,
                color=colors[i],
                line=dict(width=0.5, color='rgb(40,40,40)'),
                sizemode='area'
            ),
            name='topo {0} with data size {1}'.format(i,len(vertex)))
        communities.append(community)

    layout = dict(
        title='Terrorist communities -- %s <br>(Click legend to toggle traces)'%plotFea,
        showlegend=True,
        geo=dict(
            scope='',
            # projection=dict(type='albers usa'),
            showland=True,
            landcolor='rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

    fig = dict(data=communities, layout=layout)
    py.iplot(fig, validate=False, filename='bubble-map-terrorists-communities')
    plt.show()




if __name__=='__main__':
    colors = getColorRGB('jet',880)
    print(colors)
    # plotTest()