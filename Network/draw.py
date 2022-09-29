import os
import torch
import random
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def rgb2gray(rgb):
    r,g,b = rgb[0,:,:],rgb[1,:,:],rgb[2,:,:]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    return gray

def draw_samples(samples_list, attr_list,validation=False, cmap=None, norm=None, methods=None, colorbar=True, save=False, save_file='samples.png'):
    
    r, num = len(attr_list), len(samples_list)
    clabels = list(attr_list)   
    
    if cmap is None:
        cmap = []
        for key in clabels:
            if key in ["seis"]:
                cmap.append("gray")
            else:
                cmap.append("jet")

    methods = []
    for key in clabels:
        if key in ["label", "pred"]:
            methods.append("nearest")
        else:
            methods.append("bilinear")
 
    extent = None

    fig, axs = plt.subplots(r, num, sharey=False, figsize=(17*num, 8*r))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=None)

    for j, attr in enumerate(attr_list):
        
        if norm is not None:
            norm_ = mpl.colors.Normalize(vmin=norm[j][0], vmax=norm[j][1])
        else:
            norm_ = None
                    
        for i in range(num): 
            section = samples_list[i][attr].squeeze()     
            if attr in ["label", "pred"]: 
                if(validation):
                    vmin, vmax = 0, 1
                else:
                    vmin, vmax = 0, 2
            else:
                vmin, vmax = None, None
#                 vmin, vmax = -2,2 # gaohui set

            if (r == 1) & (num > 1):
                im = axs[i].imshow(section, aspect='auto', extent=extent, cmap=cmap[j], vmin=vmin, vmax=vmax, 
                                   interpolation=methods[j], norm=norm_)
            elif (r > 1) & (num == 1):
                im = axs[j].imshow(section, aspect='auto', extent=extent, cmap=cmap[j], vmin=vmin, vmax=vmax,  
                                   interpolation=methods[j], norm=norm_)                    
            elif (r == 1) & (num == 1):
                im = axs.imshow(section, aspect='auto', extent=extent, cmap=cmap[j], vmin=vmin, vmax=vmax,  
                                   interpolation=methods[j], norm=norm_)
            else:
                im = axs[j, i].imshow(section, aspect='auto', extent=extent, cmap=cmap[j], vmin=vmin, vmax=vmax,  
                                   interpolation=methods[j], norm=norm_)

            if (r == 1) & (num > 1):
                axs[i].set_xlabel('X', fontsize=18)
                axs[i].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[i].set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs[i], pad=0.02)
            elif (r > 1) & (num == 1):
                axs[j].set_xlabel('X', fontsize=18)
                axs[j].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[j].set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs[j], pad=0.02)
            elif (r == 1) & (num == 1):
                axs.set_xlabel('X', fontsize=18)
                axs.set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs.set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs, pad=0.02)
            else:
                axs[j, i].set_xlabel('X', fontsize=18)
                axs[j, i].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[j, i].set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs[j, i], pad=0.02)
    if save:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def draw_samples_sp(samples_list, attr_list, cmap=None, norm=None, methods=None, colorbar=True, save=False, mask=False,save_file='samples.png'):
    
    r, num = len(attr_list), len(samples_list)
    clabels = list(attr_list)   
    
    if cmap is None:
        cmap = []
        for key in clabels:
            if key in ["seis"]:
                cmap.append("gray")
            else:
                cmap.append("jet")

    methods = []
    for key in clabels:
        if key in ["label", "pred"]:
            methods.append("nearest")
        else:
            methods.append("bilinear")
 
    extent = None

    fig, axs = plt.subplots(r, num, sharey=False, figsize=(17*num, 8*r))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=None)

    for j, attr in enumerate(attr_list):
        
        if norm is not None:
            norm_ = mpl.colors.Normalize(vmin=norm[j][0], vmax=norm[j][1])
        else:
            norm_ = None
                    
        for i in range(num): 
            section = samples_list[i][attr].squeeze()     
            if attr in ["label", "pred"]:    
                vmin, vmax = 0, 1
                if mask == True:
                    Mask = samples_list[i]["mask"].squeeze() 
                    section = np.where(Mask==0,np.nan,section)
            else:
#                 vmin, vmax = None, None
                vmin, vmax = -3,3 # gaohui set

            if (r == 1) & (num > 1):
                im = axs[i].imshow(section, aspect='auto', extent=extent, cmap=cmap[j], vmin=vmin, vmax=vmax, 
                                   interpolation=methods[j], norm=norm_)
            elif (r > 1) & (num == 1):
                im = axs[j].imshow(section, aspect='auto', extent=extent, cmap=cmap[j], vmin=vmin, vmax=vmax,  
                                   interpolation=methods[j], norm=norm_)                    
            elif (r == 1) & (num == 1):
                im = axs.imshow(section, aspect='auto', extent=extent, cmap=cmap[j], vmin=vmin, vmax=vmax,  
                                   interpolation=methods[j], norm=norm_)
            else:
                im = axs[j, i].imshow(section, aspect='auto', extent=extent, cmap=cmap[j], vmin=vmin, vmax=vmax,  
                                   interpolation=methods[j], norm=norm_)

            if (r == 1) & (num > 1):
                axs[i].set_xlabel('X', fontsize=18)
                axs[i].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[i].set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs[i], pad=0.02)
            elif (r > 1) & (num == 1):
                axs[j].set_xlabel('X', fontsize=18)
                axs[j].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[j].set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs[j], pad=0.02)
            elif (r == 1) & (num == 1):
                axs.set_xlabel('X', fontsize=18)
                axs.set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs.set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs, pad=0.02)
            else:
                axs[j, i].set_xlabel('X', fontsize=18)
                axs[j, i].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[j, i].set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs[j, i], pad=0.02)
    if save:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def draw_img(img, msk=None, cmap="jet", method="bilinear"):
    plt.imshow(img,cmap=cmap, interpolation=method)
    if msk is not None:
        plt.imshow(msk, alpha=0.4, cmap='jet', interpolation='nearest')  
    plt.colorbar(fraction=0.023,pad=0.02) 
    
def draw_slice(volume, x_slice, y_slice, z_slice, cmap='jet',clab=None):
    if len(volume.shape) > 3:
        volume = volume.squeeze()
    z, y, x = volume.shape
    cmin=np.min(volume)
    cmax=np.max(volume)
    
    if clab is None:
        showscale = False
    else:
        showscale = True
        
    # x-slice
    yy = np.arange(0, y, 1)
    zz = np.arange(0, z, 1)
    yy,zz = np.meshgrid(yy,zz)
    xx = x_slice * np.ones((y, z)).T
    vv = volume[:,:,x_slice]
    fig = go.Figure(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=showscale,
        colorbar={"title":clab, 
                  "title_side":'right',
                  "len": 0.8,
                  "thickness": 8,
                  "xanchor":"right"}))

    # y-slice
    xx = np.arange(0, x, 1)
    zz = np.arange(0, z, 1)
    xx,zz = np.meshgrid(xx,zz)
    yy = y_slice * np.ones((x, z)).T
    vv = volume[:,y_slice,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    # z-slice
    xx = np.arange(0, x, 1)
    yy = np.arange(0, y, 1)
    xx,yy = np.meshgrid(xx,yy)
    zz = z_slice * np.ones((x, y)).T
    vv = volume[z_slice,:,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    fig.update_layout(
            height=400,
            width=600,
            scene = {
            "xaxis": {"nticks": 5, "title":"Corssline"},
            "yaxis": {"nticks": 5, "title":"Inline"},
            "zaxis": {"nticks": 5, "autorange":'reversed', "title":"Sample"},
            'camera_eye': {"x": 1.25, "y": 1.25, "z": 1.25},
            'camera_up': {"x": 0, "y": 0, "z": 1},
            "aspectratio": {"x": 1, "y": 1, "z": 1.05}
            },
            margin=dict(t=0, l=0, b=0))
    fig.show()
    
def draw_slice_surf(volume, x_slice, y_slice, z_slice, cmap='jet', color='cyan', clab=None, isofs=None, surfs=None):
    if len(volume.shape) > 3:
        volume = volume.squeeze()
    nz, ny, nx = volume.shape
    cmin=np.min(volume)
    cmax=np.max(volume)
    
    if clab is None:
        showscale = False
    else:
        showscale = True
      
    fig = go.Figure()  
    
    # surf
    if surfs is not None:
        for surf in surfs:
            xx,yy,zz = [],[],[]
            for ix in range(0,nx,2):
                for iy in range(0,ny,2):
                    if surf[iy][ix]>0 and surf[iy][ix]<nz-1:
                        xx.append(ix)
                        yy.append(iy)
                        zz.append(surf[iy][ix])
            obj = {}        
            obj.update({"type": "mesh3d",
                        "x": xx,
                        "y": yy,
                        "z": zz,
                        "color": color,
                        "opacity": 0.5})         
            fig.add_trace(obj)
            
    # iso-surf
    if isofs is not None:
        for isof in isofs:
            obj = {}
            verts, faces, normals, values = measure.marching_cubes(volume.transpose(2,1,0), isof, step_size=2)
            obj.update({"type": "mesh3d",
                        "x": verts[:, 0],
                        "y": verts[:, 1],
                        "z": verts[:, 2],
                        "i": faces[:, 0],
                        "j": faces[:, 1],
                        "k": faces[:, 2],
                        "intensity": np.ones(len(verts[:, 0])) * isof,
                        "colorscale": cmap,
                        "showscale": False,
                        "cmin": cmin,
                        "cmax": cmax})
            fig.add_trace(obj)
    
    # x-slice
    yy = np.arange(0, ny, 1)
    zz = np.arange(0, nz, 1)
    yy,zz = np.meshgrid(yy,zz)
    xx = x_slice * np.ones((ny, nz)).T
    vv = volume[:,:,x_slice]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=showscale,
        colorbar={"title":clab, 
                  "title_side":'right',
                  "len": 0.8,
                  "thickness": 8,
                  "xanchor":"right"}))

    # y-slice
    xx = np.arange(0, nx, 1)
    zz = np.arange(0, nz, 1)
    xx,zz = np.meshgrid(xx,zz)
    yy = y_slice * np.ones((nx, nz)).T
    vv = volume[:,y_slice,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    # z-slice
    xx = np.arange(0, nx, 1)
    yy = np.arange(0, ny, 1)
    xx,yy = np.meshgrid(xx,yy)
    zz = z_slice * np.ones((nx, ny)).T
    vv = volume[z_slice,:,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    fig.update_layout(
            height=400,
            width=600,
            scene = {
            "xaxis": {"nticks": 5, "title":"Corssline"},
            "yaxis": {"nticks": 5, "title":"Inline"},
            "zaxis": {"nticks": 5, "autorange":'reversed', "title":"Sample"},
            'camera_eye': {"x": 1.25, "y": 1.25, "z": 1.25},
            'camera_up': {"x": 0, "y": 0, "z": 1},
            "aspectratio": {"x": 1, "y": 1, "z": 1.05}
            },
            margin=dict(t=0, l=0, b=0))
    fig.show()