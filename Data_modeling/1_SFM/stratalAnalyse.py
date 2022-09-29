##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to analyse stratigraphic sequences from Badlands outputs.
"""

import os
import math
import h5py
import errno
import numpy as np
import pandas as pd
from cmocean import cm
import colorlover as cl
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
import scipy.ndimage.filters as filters
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter

import plotly
from plotly import tools
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def readSea(seafile):
    """
    Plot sea level curve.
    Parameters
    ----------
    variable: seafile
        Absolute path of the sea-lelve data.
    """

    df=pd.read_csv(seafile, sep=r'\s+',header=None)
    SLtime,sealevel = df[0],df[1]

    return SLtime,sealevel

def viewData(x0 = None, y0 = None, width = 800, height = 400, linesize = 3, color = '#6666FF',
             xlegend = 'xaxis', ylegend = 'yaxis', title = 'view data'):
    """
    Plot multiple data on a graph.
    Parameters
    ----------
    variable: x0, y0
        Data for plot
    variable: width, height
        Figure width and height.
    variable: linesize
        Requested size for the line.
    variable: color
        
    variable: xlegend
        Legend of the x axis.
    variable: ylegend
        Legend of the y axis.
    variable: title
        Title of the graph.
    """
    trace = Scatter(
        x=x0,
        y=y0,
        mode='lines',
        line=dict(
            shape='linear',
            color = color,
            width = linesize
        ),
        fill=None
    )

    layout = dict(
            title=title,
            font=dict(size=10),
            width=width,
            height=height,
            showlegend = False,
            xaxis=dict(title=xlegend,
                       ticks='outside',
                       zeroline=False,
                       showline=True,
                       mirror='ticks'),
            yaxis=dict(title=ylegend,
                       ticks='outside',
                       zeroline=False,
                       showline=True,
                       mirror='ticks')
            )

    fig = Figure(data=[trace], layout=layout)
    plotly.offline.iplot(fig)

    return

def depthID(cs = None, sealevel = None, depthIDs = None):
    """
    Calculate the position of different depositional environments for Wheeler diagram.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: sealevel
        The value of sea level through time.
    variable: envIDs
        Range of water depth of each depostional environment.
    """
    enviID = np.zeros(len(depthIDs))
    for i in range(len(depthIDs)):
        enviID[i] = np.amax(np.where((cs.secDep[cs.nz-1]) > (sealevel - depthIDs[i]))[0])

    return enviID

def viewSection(width = 800, height = 400, cs = None, dnlay = None,
                rangeX = None, rangeY = None, linesize = 3, title = 'Cross section'):
    """
    Plot multiple cross-sections data on a graph.
    Parameters
    ----------
    variable: width
        Figure width.
    variable: height
        Figure height.
    variable: cs
        Cross-sections dataset.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    nlay = len(cs.secDep)
    colors = cl.scales['9']['div']['BrBG']
    hist = cl.interp( colors, nlay )
    colorrgb = cl.to_rgb( hist )

    trace = {}
    data = []

    trace[0] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='linear',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[0])

    for i in range(1,nlay-1,dnlay):
        trace[i] = Scatter(
            x=cs.dist,
            y=cs.secDep[i],
            mode='lines',
            line=dict(
                shape='linear',
                width = linesize,
                color = 'rgb(0,0,0)'
            ),
            opacity=0.5,
            fill='tonexty',
            fillcolor=colorrgb[i]
        )
        data.append(trace[i])

    trace[nlay-1] = Scatter(
        x=cs.dist,
        y=cs.secDep[nlay-1],
        mode='lines',
        line=dict(
            shape='linear',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        ),
        fill='tonexty',
        fillcolor=colorrgb[nlay-1]
    )
    data.append(trace[nlay-1])

    trace[nlay] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='linear',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[nlay])

    if rangeX is not None and rangeY is not None:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height,
                showlegend = False,
                xaxis=dict(title='distance [m]',
                            range=rangeX,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks'),
                yaxis=dict(title='elevation [m]',
                            range=rangeY,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks')
        )
    else:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height
        )
    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return

def viewStrata(width = 9, height = 4, cs = None, enviID = None, dnlay = None, colors = None,
                      rangeX = None, rangeY = None, linesize = 0.3, title = 'Cross section'):
    """
    Plot stratal stacking pattern colored by deposition depth.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: enviID
        Positions for each depositional environment on the cross-section.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: colors
        Colors for different ranges of water depth (i.e. depositional environments).
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    fig = plt.figure(figsize = (width,height))
    plt.rc("font", size=10)

    ax = fig.add_subplot(111)
    layID = []
    p = 0
    xi00 = cs.dist
    for i in range(0,cs.nz+1,dnlay):
        if i == cs.nz:
            i = cs.nz-1
        layID.append(i)
        if len(layID) > 1:
            for j in range(enviID.shape[0]-1):
                ID1=np.transpose(enviID)[p][j]
                ID2=np.transpose(enviID)[p][j+1]
                for k in range(int(ID1),int(ID2)):
                    ax.fill_between([xi00[k],xi00[k+1]], [-cs.secDep[layID[p-1]][k], -cs.secDep[layID[p-1]][k+1]], color=colors[j])
        ax.fill_between(xi00, -cs.secDep[layID[p]], 0, color='white')
        ax.plot(xi00,-cs.secDep[layID[p]],'-',color='k',linewidth=linesize)
        p=p+1
    
    plt.xlim( rangeX )
    plt.xlabel('Distance (m)')
    plt.ylim( rangeY )
    plt.ylabel('Depth (m)')
    plt.title(title)

    return

def viewWheeDiag(width = 9, height = 4, cs = None, enviID = None, dnlay = None, time = None, colors = None,
                      rangeX = None, rangeY = None, dt_height = 0.3, title = 'Wheeler Diagram'):
    """
    Plot stratal stacking pattern colored by deposition depth.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: enviID
        Positions for each depositional environment on the cross-section.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: time
        Layer time.
    variable: colors
        Colors for different ranges of water depth (i.e. depositional environments).
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: dt_height
        Requested grid space for the time line.
    variable: title
        Title of the graph.
    """
    
    fig = plt.figure(figsize = (width,height))
    plt.rc("font", size=10)

    ax = fig.add_subplot(111)
    patch_handles = []
    for i, d in enumerate(enviID):
        patch_handles.append(plt.barh(time,d,color=colors[i],align='edge',left=d, height=dt_height, edgecolor = "none"))
    for j in range(len(time)): 
        plt.axhline(time[j], color='k', linewidth=0.1)
    plt.plot(cs.dist[cs.shoreID], time,'grey',linewidth=2)
    plt.plot(cs.dist[cs.shoreID], time,'ko',markersize=3)
    plt.xlim( rangeX )
    plt.xlabel('Distance (m)')
    plt.ylim( rangeY )
    plt.ylabel('Time (Myr)')
    plt.title(title)
    
    return

def viewStack(width = 4, height = 6, layTh = None, colorFill = None):
    """
    Plot wheeler diagram colored by deposition environment on a graph.
    Parameters
    ----------
    variable: width, height
        Figure width and height.
    variable: layTh
        Layer thickness for each wells.
    variable: colorFill
        Layer environment color for each wells.
    """

    fig = plt.figure(figsize = (width,height))
    plt.rc("font", size=10)
    
    ax = fig.add_axes([0.2,0.06,0.82,0.91])

    data = layTh
    for k in range(len(data)):
        bottom = np.cumsum(data[k], axis=0)
        colors = np.fliplr([colorFill[k]])[0]
        plt.bar(2*k, data[k][0], color = 'w', edgecolor='lightgrey', hatch = '/')
        for j in range(1, data[k].shape[0]):
            plt.bar(2*k, data[k][j], color=colors[j], edgecolor='black', bottom=bottom[j-1])

    plt.ylabel('Elevation (m)')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.axes.get_xaxis().set_visible(False)
    ax.tick_params(axis='both')
    ax.yaxis.set_ticks_position('left')

    return

def build_ShoreTrajectory(x, y, grad, sl, nbout, cTC='rgba(56,110,164,0.8)', cDRC='rgba(60,165,67,0.8)',
                          cARC='rgba(112,54,127,0.8)', cSTC='rgba(252,149,7,0.8)'):
    """
    Automatic delimitation of shoreline trajectory classes.
    Parameters
    ----------
    variable: x
        display steps
    variable: y
        shoreline position
    variable: grad
        shoreline position gradient
    variable: sl
        sealevel position
    variable: nbout
        number of output steps

    color schema for different classes
    """

    # Find intersection between line zero and the shoreline gradient trajectory
    ids = np.argwhere(np.diff(np.sign(grad - np.zeros(len(grad)))) != 0).reshape(-1) + 0
    # Number of points to consider
    nbclass = len(ids)

    # Check if there are still some points after the last intersection
    final = False
    if ids[-1]<len(grad):
        nbclass += 1
        final = True
    # Build the color list
    STcolors_ST = []

    ci0 = 0
    i0 = 0
    for k in range(nbclass):
        if k == nbclass-1:
            if not final:
                exit
            else:
                i1 = nbout
                ci1 = nbout
                i2 = -1
                sl1 = sl[i0]
                sl2 = sl[-1]
        else:
            i1 = ids[k]
            ci1 = int(x[ids[k]])
            i2 = ids[k]-1
            sl1 = sl[i0]
            sl2 = sl[ids[k]-1]
        if grad[i2] < 0:
            for p in range(ci0,ci1):
                STcolors_ST.append(cTC)
        elif grad[i2] > 0 and sl1 >= sl2:
            for p in range(ci0,ci1):
                STcolors_ST.append(cDRC)
        elif grad[i2] > 0 and sl1 < sl2:
            for p in range(ci0,ci1):
                STcolors_ST.append(cARC)
        else:
            for p in range(ci0,ci1):
                STcolors_ST.append(cSTC)
        if k < nbclass-1:
            i0 = ids[k]
            ci0 = int(x[ids[k]])

    return STcolors_ST

def build_AccomSuccession(x, y, grad, nbout, cR='rgba(51,79,217,0.8)', cPA='rgba(252,149,7,0.8)',
                          cAPD='rgba(15,112,2,0.8)'):
    """
    Automatic delimitation of accommodation succession sequence sets.
    Parameters
    ----------
    variable: x
        display steps
    variable: y
        AS curve
    variable: grad
        shoreline position gradient
    variable: nbout
        number of output steps

    color schema for different classes
    """

    # Find intersection between line zero and the AS curve
    ids1 = np.argwhere(np.diff(np.sign(y - np.zeros(len(y)))) != 0).reshape(-1) + 0
    # Find intersection between line zero and the AS gradient
    ids2 = np.argwhere(np.diff(np.sign(grad - np.zeros(len(y)))) != 0).reshape(-1) + 0
    # Combine ids together
    ids = np.concatenate((ids1,ids2))
    ids.sort(kind='mergesort')

    # Number of points to consider
    nbclass = len(ids)

    # Check if there are still some points after the last intersection
    final = False
    if ids[-1]<len(grad):
        nbclass += 1
        final = True

    # Build the color list
    STcolors_AS = []

    ci0 = 0
    i0 = 0
    for k in range(nbclass):
        if k == nbclass-1:
            if not final:
                exit
            else:
                i1 = nbout
                ci1 = nbout
                i2 = -1
        else:
            i1 = ids[k]
            ci1 = int(x[ids[k]])
            i2 = ids[k]-1
        if y[i2-1] >= 0:
            for p in range(ci0,ci1):
                STcolors_AS.append(cR)
        elif y[i2-1] < 0 and grad[i2-1] >= 0:
            for p in range(ci0,ci1):
                STcolors_AS.append(cPA)
        elif y[i2-1] < 0 and grad[i2-1] < 0:
            for p in range(ci0,ci1):
                STcolors_AS.append(cAPD)
        if k < nbclass-1:
            i0 = ids[k]
            ci0 = int(x[ids[k]])

    return STcolors_AS

def viewSectionST(width = 800, height = 400, cs = None, dnlay = None, colors=None,
                  rangeX = None, rangeY = None, linesize = 3, title = 'Cross section'):
    """
    Plot multiple cross-sections colored by system tracts on a graph.
    Parameters
    ----------
    variable: width
        Figure width.
    variable: height
        Figure height.
    variable: cs
        Cross-sections dataset.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: colors
        System tract color scale.
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    nlay = len(cs.secDep)

    trace = {}
    data = []

    trace[0] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='linear',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[0])

    for i in range(1,nlay-1,dnlay):
        trace[i] = Scatter(
            x=cs.dist,
            y=cs.secDep[i],
            mode='lines',
            line=dict(
                shape='linear',
                width = linesize,
                color = 'rgb(0,0,0)'
            ),
            opacity=0.5,
            fill='tonexty',
            fillcolor=colors[i]
        )
        data.append(trace[i])

    trace[nlay-1] = Scatter(
        x=cs.dist,
        y=cs.secDep[nlay-1],
        mode='lines',
        line=dict(
            shape='linear',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        ),
        fill='tonexty',
        fillcolor=colors[0]
    )
    data.append(trace[nlay-1])

    trace[nlay] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='linear',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[nlay])

    if rangeX is not None and rangeY is not None:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height,
                showlegend = False,
                xaxis=dict(title='distance [m]',
                            range=rangeX,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks'),
                yaxis=dict(title='elevation [m]',
                            range=rangeY,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks')
        )
    else:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height
        )
    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return

class stratalSection:
    """
    Class for creating stratigraphic cross-sections from Badlands outputs.
    """

    def __init__(self, folder=None, ncpus=1):
        """
        Initialization function which takes the folder path to Badlands outputs
        and the number of CPUs used to run the simulation.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        variable: ncpus
            Number of CPUs used to run the simulation.
        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.ncpus = ncpus
        if ncpus > 1:
            raise RuntimeError('Multi-processors function not implemented yet!')

        self.x = None
        self.y = None
        self.xi = None
        self.yi = None
        self.dx = None
        self.dist = None
        self.dx = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.dep = None
        self.th = None
        self.elev = None
        self.xsec = None
        self.ysec = None
        self.secTh = []
        self.secDep = []
        self.secElev = []

        self.shoreID = []
        self.shoreID_elev = []
        self.shelfedge = []
        self.shelfedge_elev = []
        self.depoend = []
        self.depoend_elev = []
        self.accom_shore = []
        self.sed_shore = []

        return
    
    def _angle(self, data):
        """
        Compute angle of a curve.
        """
        
        dir2 = data[1:]
        dir1 = data[:-1]
        
        return np.arccos((dir1*dir2).sum(axis=1)/(np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))
    
    def buildParameters(self, CS = None, Sealevel = None, min_angle = None):
        """
        Calculate shoreline, shelf-edge, and downlap trajectories, rate of accommodation creation, rate of sedimentation.
        Parameters
        ----------
        variable : CS
            Cross-section dataset.
        variable : Sealevel
            Sealevel value for each stratigraphic layer.
        variable : min_angle
            Critical slope to calculate shelf-edge position.
        """

        for i in range(len(CS)):
            self.shoreID.append(np.amax(np.where(CS[i].secDep[CS[i].nz-1]>=Sealevel[i])[0]))
            self.shoreID_elev.append(CS[i].secDep[CS[i].nz-1][int(self.shoreID[i])])
            self.depoend.append(np.amax(np.where(sum(np.asarray(CS[i].secTh[0:CS[i].nz]), 0)>0.1)[0]))
            self.depoend_elev.append(CS[i].secDep[CS[i].nz-1][int(self.depoend[i])])
            # shelf-edge
            xdata = CS[i].dist[int(self.shoreID[i]):int(self.depoend[i])]
            ydata = CS[i].secDep[CS[i].nz-1][int(self.shoreID[i]):int(self.depoend[i])]
            curve = np.vstack((xdata,ydata))
            sx, sy = np.array(curve.T).T
            theta = self._angle(np.diff(curve.T, axis=0))
            idx = np.where(theta>min_angle)[0]+1
            if(idx.shape[0]):
                self.shelfedge.append(sx[idx[0]])
                self.shelfedge_elev.append(sy[idx[0]])
            else:
                self.shelfedge.append(None)
                self.shelfedge_elev.append(None)

            # A and S at shoreline
            if i>1:
                shoreID_b = np.amax(np.where(CS[i-1].secDep[CS[i-1].nz-1]>=Sealevel[i-1])[0])
                self.accom_shore.append(Sealevel[i] - CS[i].secDep[0][shoreID_b])
                self.sed_shore.append(CS[i].secDep[CS[i].nz-1][shoreID_b] - CS[i].secDep[0][shoreID_b])

        return  

    def loadStratigraphy(self, timestep=0):
        """
        Read the HDF5 file for a given time step.
        Parameters
        ----------
        variable : timestep
            Time step to load.
        """

        for i in range(0, self.ncpus):
            df = h5py.File('%s/sed.time%s.hdf5'%(self.folder, timestep), 'r')
            #print(list(df.keys()))
            coords = np.array((df['/coords']))
            layDepth = np.array((df['/layDepth']))
            layElev = np.array((df['/layElev']))
            layThick = np.array((df['/layThick']))
            if i == 0:
                x, y = np.hsplit(coords, 2)
                dep = layDepth
                elev = layElev
                th = layThick

        self.dx = x[1]-x[0]
        self.x = x
        self.y = y
        self.nx = int((x.max() - x.min())/self.dx+1)
        self.ny = int((y.max() - y.min())/self.dx+1)
        self.nz = dep.shape[1]
        self.xi = np.linspace(x.min(), x.max(), self.nx)
        self.yi = np.linspace(y.min(), y.max(), self.ny)
        self.dep = dep.reshape((self.ny,self.nx,self.nz))
        self.elev = elev.reshape((self.ny,self.nx,self.nz))
        self.th = th.reshape((self.ny,self.nx,self.nz))

        return
    
    def loadStratigraphy_multi(self, timestep=0, nstep=None):
        """
        Read the HDF5 file for a given time step.
        Parameters
        ----------
        variable : timestep
            Time step to load.
        """

        for i in range(0, self.ncpus):
            df = h5py.File('%s/sed.time%s.hdf5'%(self.folder, timestep), 'r')
            #print(list(df.keys()))
            if timestep > 1+nstep:
                coords = np.array((df['/coords']))
                layDepth = np.array((df['/layDepth'][:,timestep-1-nstep:timestep-1]))
                layElev = np.array((df['/layElev'][:,timestep-1-nstep:timestep-1]))
                layThick = np.array((df['/layThick'][:,timestep-1-nstep:timestep-1]))
            
            if timestep < 1+nstep:
                coords = np.array((df['/coords']))
                layDepth = np.array((df['/layDepth'][:,0:timestep-1]))
                layElev = np.array((df['/layElev'][:,0:timestep-1]))
                layThick = np.array((df['/layThick'][:,0:timestep-1]))
            
            if timestep == 1:
                coords = np.array((df['/coords']))
                layDepth = np.array((df['/layDepth']))
                layElev = np.array((df['/layElev']))
                layThick = np.array((df['/layThick']))
            
            if i == 0:
                x, y = np.hsplit(coords, 2)
                dep = layDepth
                elev = layElev
                th = layThick

        self.dx = x[1]-x[0]
        self.x = x
        self.y = y
        self.nx = int((x.max() - x.min())/self.dx+1)
        self.ny = int((y.max() - y.min())/self.dx+1)
        self.nz = th.shape[1]
        self.xi = np.linspace(x.min(), x.max(), self.nx)
        self.yi = np.linspace(y.min(), y.max(), self.ny)
        self.dep = dep.reshape((self.ny,self.nx,self.nz))
        self.elev = elev.reshape((self.ny,self.nx,self.nz))
        self.th = th.reshape((self.ny,self.nx,self.nz))

        return

    def _cross_section(self, xo, yo, xm, ym, pts):
        """
        Compute cross section coordinates.
        """

        if xm == xo:
            ysec = np.linspace(yo, ym, pts)
            xsec = np.zeros(pts)
            xsec.fill(xo)
        elif ym == yo:
            xsec = np.linspace(xo, xm, pts)
            ysec = np.zeros(pts)
            ysec.fill(yo)
        else:
            a = (ym-yo)/(xm-xo)
            b = yo - a * xo
            xsec = np.linspace(xo, xm, pts)
            ysec = a * xsec + b

        return xsec, ysec

    def buildSection(self, xo = None, yo = None, xm = None, ym = None,
                    pts = 100, gfilter = 5):
        """
        Extract a slice from the 3D data set and compute the stratigraphic layers.
        Parameters
        ----------
        variable: xo, yo
            Lower X,Y coordinates of the cross-section.
        variable: xm, ym
            Upper X,Y coordinates of the cross-section.
        variable: pts
            Number of points to discretise the cross-section.
        variable: gfilter
            Gaussian smoothing filter.
        """

        if xm > self.x.max():
            xm = self.x.max()

        if ym > self.y.max():
            ym = self.y.max()

        if xo < self.x.min():
            xo = self.x.min()

        if yo < self.y.min():
            yo = self.y.min()

        xsec, ysec = self._cross_section(xo, yo, xm, ym, pts)
        self.dist = np.sqrt(( xsec - xo )**2 + ( ysec - yo )**2)
        self.xsec = xsec
        self.ysec = ysec
        for k in range(self.nz):
            # Thick
            rect_B_spline = RectBivariateSpline(self.yi, self.xi, self.th[:,:,k])
            data = rect_B_spline.ev(ysec, xsec)
            secTh = filters.gaussian_filter1d(data,sigma=gfilter)
            secTh[secTh < 0] = 0
            self.secTh.append(secTh)

            # Elev
            rect_B_spline1 = RectBivariateSpline(self.yi, self.xi, self.elev[:,:,k])
            data1 = rect_B_spline1.ev(ysec, xsec)
            secElev = filters.gaussian_filter1d(data1,sigma=gfilter)
            self.secElev.append(secElev)

            # Depth
            rect_B_spline2 = RectBivariateSpline(self.yi, self.xi, self.dep[:,:,k])
            data2 = rect_B_spline2.ev(ysec, xsec)
            secDep = filters.gaussian_filter1d(data2,sigma=gfilter)
            self.secDep.append(secDep)

        # Ensure the spline interpolation does not create underlying layers above upper ones
        topsec = self.secDep[self.nz-1]
        for k in range(self.nz-2,-1,-1):
            secDep = self.secDep[k]
            self.secDep[k] = np.minimum(secDep, topsec)
            topsec = self.secDep[k]

        return
    
    def buildSection_multi(self, xo = None, yo = None, xm = None, ym = None,
                    pts = 100, gfilter = 5, nstep = 5):
        """
        Extract a slice from the 3D data set and compute the stratigraphic layers.
        Parameters
        ----------
        variable: xo, yo
            Lower X,Y coordinates of the cross-section.
        variable: xm, ym
            Upper X,Y coordinates of the cross-section.
        variable: pts
            Number of points to discretise the cross-section.
        variable: gfilter
            Gaussian smoothing filter.
        """

        if xm > self.x.max():
            xm = self.x.max()

        if ym > self.y.max():
            ym = self.y.max()

        if xo < self.x.min():
            xo = self.x.min()

        if yo < self.y.min():
            yo = self.y.min()

        xsec, ysec = self._cross_section(xo, yo, xm, ym, pts)
        self.dist = np.sqrt(( xsec - xo )**2 + ( ysec - yo )**2)
        self.xsec = xsec
        self.ysec = ysec
            
        for k in range(self.nz):
            if (self.nz-1<0):
                k = 0
            # Thick
            rect_B_spline = RectBivariateSpline(self.yi, self.xi, self.th[:,:,k])
            data = rect_B_spline.ev(ysec, xsec)
            secTh = filters.gaussian_filter1d(data,sigma=gfilter)
            secTh[secTh < 0] = 0
            self.secTh.append(secTh)
            
            # Elev
            rect_B_spline1 = RectBivariateSpline(self.yi, self.xi, self.elev[:,:,k])
            data1 = rect_B_spline1.ev(ysec, xsec)
            secElev = filters.gaussian_filter1d(data1,sigma=gfilter)
            self.secElev.append(secElev)

            # Depth
            rect_B_spline2 = RectBivariateSpline(self.yi, self.xi, self.dep[:,:,k])
            data2 = rect_B_spline2.ev(ysec, xsec)
            secDep = filters.gaussian_filter1d(data2,sigma=gfilter)
            self.secDep.append(secDep)
        
        # Ensure the spline interpolation does not create underlying layers above upper ones
        if (self.nz-2>0):
            topsec = self.secDep[self.nz-1]
            for k in range(self.nz-2,-1,-1):
                secDep = self.secDep[k]
                self.secDep[k] = np.minimum(secDep, topsec)
                topsec = self.secDep[k]

        return
