import h5py
import os, sys
import math
import random
import numpy as np
import pandas as pd
from shutil import copy
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

cut_range1 = int(round(random.uniform(550,575),0))
cut_range2 = cut_range1 + 410
Cut_Range = cut_range2-cut_range1
Shape0 = [401, 1001, 101]
section_num = int(Shape0[0]/2)
Data_shape = [Cut_Range,101] # Data_shape[2]= cut_range2 - cut_range1
# plotOnly = True
plotOnly = False


def main(ip,folder,savedir,model_number):
    mkdir(savedir)
    gocopy3dat(folder,savedir,model_number)
    goLoadData2d(folder,savedir,model_number)
    goGetSlope2d(folder,savedir,model_number)
    goInterpolation2d(folder,savedir,model_number)
    goImpData2d(folder,savedir,model_number)
    print("**** data postprocessing is done —— model: ",ip)    

def gocopy3dat(folder,savedir,model_number):
    sealevel_name = "../1_SFM/Inputs/Models/inputs/demo"+str(model_number)+"/sealevel.csv"
    node_name = "../1_SFM/Inputs/Models/inputs/demo"+str(model_number)+"/node.csv"
    thermsub_name = "../1_SFM/Inputs/Models/inputs/demo"+str(model_number)+"/thermsub.csv"
    savepath = savedir+"inputs/"
    mkdir(savepath)
    copy(sealevel_name,savepath +"sealevel.csv")
    copy(node_name,savepath +"node.csv")
    copy(thermsub_name,savepath +"thermsub.csv")
    
def goLoadData2d(folder,savedir,model_number):
    sed = h5py.File(folder + 'sed.time50.hdf5', 'r')
    coords = np.array(sed['/coords'])
    layDepth = np.array(sed['/layDepth'])  # load layDepth data
    layPoro = np.array(sed['/layPoro'])  # load layPoro data
    layElev = np.array(sed['/layElev']) # load poleawater depth data
    x, y = np.hsplit(coords, 2)
    dx = x[1] - x[0]
    nx = int((x.max() - x.min()) / dx + 1)
    ny = int((y.max() - y.min()) / dx + 1)
    nz = layDepth.shape[1]
    depth = layDepth.reshape((ny, nx, nz))  # depth data
    poro = layPoro.reshape((ny, nx, nz))  # poro data
    elev = layElev.reshape((ny, nx, nz))  # elev data
    Dep = depth[section_num, cut_range1:cut_range2, :] #2d
    Poro = poro[section_num, cut_range1:cut_range2, :]
    Elev = elev[section_num, cut_range1:cut_range2, :]
    write_cube2d(Dep, savedir + 'Dep0.dat')
    write_cube2d(Poro, savedir + 'Poro0.dat')
    write_cube2d(Elev, savedir + 'Elev0.dat')
    ## build RGT data
    Tmax = (Data_shape[1] - 1) / 5   # RGT max value
    RGTi = [Tmax]
    for m in range(1, Data_shape[1]):
        n = Tmax - (m * 0.2)
        RGTi = np.append(RGTi, [n], axis=0)
    RGT = np.float32([[None] * Data_shape[1]] * Data_shape[0])
    for i in range(Data_shape[0]):
        RGT[i, :] = RGTi
    write_cube2d(RGT, savedir + 'RGT0.dat')
    
def goGetSlope2d(folder,savedir,model_number):
    RGT = np.fromfile(savedir + 'RGT0.dat', dtype=np.float32).reshape(Data_shape)
    Dep = np.fromfile(savedir + 'Dep0.dat', dtype=np.float32).reshape(Data_shape)
    Poro = np.fromfile(savedir + 'Poro0.dat', dtype=np.float32).reshape(Data_shape)
    Elev = np.fromfile(savedir + 'Elev0.dat', dtype=np.float32).reshape(Data_shape)
    # smoothing by using Gaussian filter
    Dep = gaussfilter1D(Dep, shape=Data_shape, sigma=3)
    # ccalculate the slope
    Slope = np.gradient(Dep, axis=0) * (-0.01)
    # smoothing by using Gaussian filter
    Slope = gaussfilter1D(Slope, shape=Data_shape, sigma=3)
    Poro = gaussfilter1D(Poro, shape=Data_shape, sigma=3)
    RGT = gaussfilter1D(RGT, shape=Data_shape, sigma=3)
    Elev = gaussfilter1D(Elev, shape=Data_shape, sigma=3)
    # save smoothed data
    write_cube2d(Dep, savedir + 'Dep.dat')
    write_cube2d(Slope, savedir + 'Slope.dat')
    write_cube2d(RGT, savedir + 'RGT.dat')
    write_cube2d(Poro, savedir + 'Poro.dat')
    write_cube2d(Elev, savedir + 'Elev.dat')

def goInterpolation2d(folder,savedir,model_number):
    # interpolate the date
    RGT = np.fromfile(savedir + 'RGT.dat', dtype=np.float32).reshape((Data_shape[0], Data_shape[1]))
    Dep = np.fromfile(savedir + 'Dep.dat', dtype=np.float32).reshape((Data_shape[0], Data_shape[1]))
    Poro = np.fromfile(savedir + 'Poro.dat', dtype=np.float32).reshape((Data_shape[0], Data_shape[1]))
    Slope = np.fromfile(savedir + 'Slope.dat', dtype=np.float32).reshape((Data_shape[0], Data_shape[1]))
    Elev = np.fromfile(savedir + 'Elev.dat', dtype=np.float32).reshape((Data_shape[0], Data_shape[1]))
    shape_interp = getshape2d(Dep)  # get the interpolation data shape 2d
    #Interpolation x
    shape0 = np.array(Dep.shape)
    shape1 = np.array(shape0)
    shape1[0] = shape1[0] * 4
    DepX = np.float32([[None] * shape1[1]] * shape1[0])
    PoroX = np.float32([[None] * shape1[1]] * shape1[0])
    RGTX = np.float32([[None] * shape1[1]] * shape1[0])
    SlopeX = np.float32([[None] * shape1[1]] * shape1[0])
    ElevX = np.float32([[None] * shape1[1]] * shape1[0])
    PointsX = np.arange(0, shape0[0], 1)
    x_range = np.array([0, shape0[0]])
    for k in range(shape0[1]):
        DepX[:, k] = discrete_point_interp_1D(PointsX, Dep[:, k], inshape_range=x_range, inv=0.25, method="linear")
        PoroX[:, k] = discrete_point_interp_1D(PointsX, Poro[:, k], inshape_range=x_range,inv=0.25, method="linear")
        RGTX[:, k] = discrete_point_interp_1D(PointsX, RGT[:, k], inshape_range=x_range,inv=0.25, method="linear")
        SlopeX[:, k] = discrete_point_interp_1D(PointsX, Slope[:, k], inshape_range=x_range,inv=0.25, method="linear")
        ElevX[:, k] = discrete_point_interp_1D(PointsX, Elev[:, k], inshape_range=x_range,inv=0.25, method="linear")
    
    # Make sure the depth direction increases monotonically
    for i in range(shape1[0]):
        for k in range(1, shape1[1]):
            if (DepX[i][k] <= DepX[i][k - 1]):
                DepX[i][k] = DepX[i][k - 1] + 0.001
    
    #Interpolation z
    maxDep = int(Dep.max()) + 1
    minDep = int(Dep.min()) - 1
    shape3 = np.array(shape1)
    shape3[1] = maxDep - minDep
    PoroXZ = np.float32([[None] * shape3[1]] * shape3[0])
    RGTXZ = np.float32([[None] * shape3[1]] * shape3[0])
    SlopeXZ = np.float32([[None] * shape3[1]] * shape3[0])
    ElevXZ = np.float32([[None] * shape3[1]] * shape3[0])
    xz_range = np.array([minDep, maxDep])
    for i in range(shape1[0]):
        PoroXZ[i, :] = discrete_point_interp_1D(DepX[i, :], PoroX[i, :], xz_range,inv=1, method='nearest')
        RGTXZ[i, :] = discrete_point_interp_1D(DepX[i, :], RGTX[i, :], xz_range,inv=1, method="linear")
        SlopeXZ[i, :] = discrete_point_interp_1D(DepX[i, :], SlopeX[i, :], xz_range,inv=1, method="linear")
        ElevXZ[i, :] = discrete_point_interp_1D(DepX[i, :], ElevX[i, :], xz_range,inv=1, method="linear")
    write_cube2d(np.flip(PoroXZ,1)[5:1605,:], savedir + 'Poro_img0.dat')
    write_cube2d(np.flip(RGTXZ,1)[5:1605,:], savedir + 'RGT_img0.dat')
    write_cube2d(np.flip(SlopeXZ,1)[5:1605,:], savedir + 'Slope_img0.dat')
    write_cube2d(np.flip(ElevXZ,1)[5:1605,:], savedir + 'Elev_img0.dat')

def goImpData2d(folder,savedir,model_number):
    Dep = np.fromfile(savedir + 'Dep.dat', dtype=np.float32).reshape((Data_shape[0], Data_shape[1]))
    shape_interp = getshape2d(Dep)
    shape_interp[0] = 1600
    # get the impedance data from porosity data
    # creat volume mask
    rgt = np.fromfile(savedir + "RGT_img0.dat", dtype=np.float32).reshape(shape_interp)
    Volmask = np.full(shape_interp, -1, dtype=np.float32)
    Volmask[np.isnan(rgt)] = 0
    Volmask = np.where(Volmask != 0, 1, Volmask)
    write_cube2d(Volmask, savedir + 'Volmask_2d.dat')
    # (Carcione.JM et al, CSEG Recorder, 2002)
    Kma = 15.45
    Uma = 13.48
    rma = 2.65
    Kfl = 2.25
    rfl = 1.040
    #read porosity data
    P = np.fromfile(savedir + "Poro_img0.dat", dtype=np.float32)
    Beta = 1 - ((1 - P) ** (3 / (1 - P)))
    M = 1 / (((Beta - P) / Kma) + (P / Kfl))
    roub = ((1 - P) * rma) + (P * rfl)
    Kfm = (Kma * (1 - Beta)) + (Beta * Beta * M)
    Ufm = Uma * (1 - Beta)
    Vp = np.sqrt(((Kfm + (4 / 3) * Ufm) / roub))
    Vs = np.sqrt((Ufm / roub))
    Zp = roub * Vp
    Zs = roub * Vs
    ZP = Zp.reshape(shape_interp)
    ZS = Zs.reshape(shape_interp)
    
    Poro = P.reshape(shape_interp)
    roub = roub.reshape(shape_interp)
    Vp = Vp.reshape(shape_interp)
    Vs = Vs.reshape(shape_interp)
    # save ZP and ZS
    write_cube2d(Vp, savedir + 'Vp_2d.dat')
    write_cube2d(Vs, savedir + 'Vs_2d.dat')
    write_cube2d(ZP, savedir + 'ZP_2d.dat')
    write_cube2d(ZS, savedir + 'ZS_2d.dat')
    write_cube2d(roub, savedir + 'Rho_2d.dat')

    # stratigraphic volume mask
    Poro = Poro * Volmask
    ZP = ZP * Volmask
    ZS = ZS * Volmask
    Vp = Vp * Volmask
    Vs = Vs * Volmask
    roub = roub * Volmask

    Poro = np.where(Volmask == 0, np.nan, Poro)
    ZP = np.where(Volmask == 0, np.nan, ZP)
    ZS = np.where(Volmask == 0, np.nan, ZS)
    Vp = np.where(Volmask == 0, np.nan, Vp)
    Vs = np.where(Volmask == 0, np.nan, Vs)
    roub = np.where(Volmask == 0, np.nan, roub)

    write_cube2d(roub, savedir + 'Rho_vol_2d.dat')
    write_cube2d(Vp, savedir + 'Vp_vol_2d.dat')
    write_cube2d(Vs, savedir + 'Vs_vol_2d.dat')
    # save ZP and ZS only have value in stratigraphic place
    write_cube2d(Poro, savedir + 'Poro_vol_2d.dat')
    write_cube2d(ZP, savedir + 'ZP_vol_2d.dat')
    write_cube2d(ZS, savedir + 'ZS_vol_2d.dat')     
    
def write_cube2d(data, path):
    data = np.transpose(data, [0, 1]).astype(np.single)
    data.tofile(path)    

def gaussfilter1D(data, shape, sigma=1):
    # gaussfilter 1D
    Data = np.float32([[None] * shape[1]] * shape[0])
    for k in range(shape[1]):
        Data[:, k] = gaussian_filter(data[:, k], sigma=sigma).astype(float)
    return Data    

def getshape2d(data):
    maxDep = int(data.max()) + 1
    minDep = int(data.min()) - 1
    shape_interp = np.array(data.shape)
    shape_interp[0] = shape_interp[0] * 4
    shape_interp[1] = maxDep - minDep
    return shape_interp    

def discrete_point_interp_1D(points, values, inshape_range, inv=0.25, method='linear'):
    grid_x = np.meshgrid(np.arange(inshape_range[0],inshape_range[1], inv))
    grid_img = griddata(points, values, grid_x, method=method)  # method <linear、cubic、nearest>
    return grid_img    

def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

#######################################################################################################################
if __name__ == "__main__":
    for i in range(1):
        model_number = str(i)
        folder = '../1_SFM/Inputs/Models/results/demo' + model_number + '_output/h5/'
        savedir = './processed_models/model'+ model_number +'/'
        mkdir(savedir)
        main(model_number,folder, savedir, model_number)

