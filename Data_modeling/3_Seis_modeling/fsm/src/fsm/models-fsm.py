import math
from utils import*

#read files
Zpfile = "ZP_2d" 
Vpfile = "Vp_2d"
Rgtfile = "RGT_img0"
Maskfile = "Volmask_2d"
Zsvolfile = "ZS_vol_2d"
Vsvolfile = "Vs_vol_2d"
Elevfile = "Elev_img0"
Porovolfile = "Poro_vol_2d"
Rhovolfile = "Rho_vol_2d"
Slopefile = "Slope_img0"

#output files
ptnewfile = "zpvol" # p Impedance in times
rtnewfile = "rpvol" # reflection parameters in times
spnewfile = "sp" # seismic data without adding noise in times
spnsnewfile = "spns" # seismic data with noise in times
rgtnewfile = "rgt" # rgt in times
masknewfile = "mask" # mask in times
zsvolnewfile = "zsvol"
vsvolnewfile = "vsvol"
elevnewfile = "elev"
porovolnewfile = "porovol"
rhovolnewfile = "rhovol"
slopenewfile = "slope"

def main(args):
  setupForSubset("models")
  shape=readTxt("models_shape.txt")
  for ip in range(1):
      n1,n2 = int(shape[ip][0]),1600
      goModel(ip,n1,n2)
      gofilter(ip)
      print("models- <"+str(ip)+"> is done")

def goModel(ip,n1,n2):
   # Forward Stratigraphic Modelling
   n3 = 100
   s1,s2,s3 = Sampling(n1),Sampling(n2),Sampling(n3)
   fs = ForwardStratigraphicModeller(s1,s2,s3)
    
   rgt = readImageclinoL2d(Rgtfile,ip,n1,n2) 
   zp = readImageclinoL2d(Zpfile,ip,n1,n2) 
   vp = readImageclinoL2d(Vpfile,ip,n1,n2)
   mask = readImageclinoL2d(Maskfile,ip,n1,n2)
   zsvol = readImageclinoL2d(Zsvolfile,ip,n1,n2)
   vsvol = readImageclinoL2d(Vsvolfile,ip,n1,n2) 
   elev = readImageclinoL2d(Elevfile,ip,n1,n2) 
   porovol = readImageclinoL2d(Porovolfile,ip,n1,n2) 
   rhovol = readImageclinoL2d(Rhovolfile,ip,n1,n2)
   slope = readImageclinoL2d(Slopefile,ip,n1,n2)
    
   rgf = RecursiveGaussianFilterP(1)
   rgf.apply00(zp,zp)
   vp = mul(vp,1000)  #convert to m/s
   dtc = Depth2TimeConverter()
   dt = 0.002 #TBD
   nt = 256
   dz = 1
   rd = Random()
   fpeak = 40 + rd.nextInt(21)  #random fequency (40~60) #TBD
   st = Sampling(nt,dt,0)
   sz = Sampling(n1,dz,0)
   rp = dtc.refFromImp2d(zp) # calculate the reflection parameters before transform
   vpt,zpt,rt,rgtt,mt,zst,vst,elevt,porot,rhot,slopet = dtc.depth2Time2D(st,sz,vp,[vp,zp,rp,rgt,mask,zsvol,vsvol,elev,porovol,rhovol,slope])
   zpt = mul(zpt,mt) # p Impedance d2t add mask
   rt = mul(rt,mt) # reflection paramater d2t add mask
   sp = dtc.addWavelet2d(dt,fpeak,rt)
   sp = mul(sp,mt)
    # add noise
   rd = Random()
   index = rd.nextInt(6) 
   files = ["F3_ns4","F3_ns6","xm_ns4","xm_ns6","al_ns4","al_ns6"]
   nsfile = files[index]
   noise = readImageNoisedtL(nsfile,index) 
   spns = fs.addRealNoise2d(noise,sp) # add noise
   spns=mul(spns,mt) # spns add mask

#output files
   writeImageclinoL(ptnewfile,zpt,ip) 
   writeImageclinoL(rtnewfile,rt,ip)
   writeImageclinoL(spnewfile,sp,ip)
   writeImageclinoL(spnsnewfile,spns,ip)

   writeImageclinoL(rgtnewfile,rgtt,ip)
   writeImageclinoL(masknewfile,mt,ip)
   writeImageclinoL(zsvolnewfile,zst,ip)
   writeImageclinoL(vsvolnewfile,vst,ip)
   writeImageclinoL(porovolnewfile,porot,ip)
   writeImageclinoL(rhovolnewfile,rhot,ip)
   writeImageclinoL(slopenewfile,slopet,ip)

# calculate the u1,u2 of the structural smoothing layer
def gofilter(ip):
   n1 = 256
   n2 = 1600
   n3 = 100
   s1,s2,s3 = Sampling(n1),Sampling(n2),Sampling(n3)
   fs = ForwardStratigraphicModeller(s1,s2,s3)
   sp = readImageclinoL2d(spnsnewfile,ip,n1,n2)
#    spnew = fs.upsample_2d(4,4,sp)
   u1 = zerofloat(n1,n2)
   u2 = zerofloat(n1,n2)
   theta = zerofloat(n1,n2)
   v1 = zerofloat(n1,n2)
   v2 = zerofloat(n1,n2)
   eu = zerofloat(n1,n2)
   ev = zerofloat(n1,n2)
   el = zerofloat(n1,n2)
   lof = LocalOrientFilter(64.0,16.0)
   lof.apply(sp,theta,u1,u2,v1,v2,eu,ev,el) 

   u1_4times = fs.upsample_2d(4,4,u1)
   u2_4times = fs.upsample_2d(4,4,u2)
   u1_16times = fs.upsample_2d(16,16,u1)
   u2_16times = fs.upsample_2d(16,16,u2)

   u1file_4times = "u1_4times"
   u2file_4times = "u2_4times"
   u1file_16times = "u1_16times"
   u2file_16times = "u2_16times" 
    
   writeImageclinoL(u1file_4times,u1_4times,ip)
   writeImageclinoL(u2file_4times,u2_4times,ip)
   writeImageclinoL(u1file_16times,u1_16times,ip)
   writeImageclinoL(u2file_16times,u2_16times,ip)
    
    
def readTxt(fname):
    k = 0
    fr = open(fname)
    data = fr.readlines()
    logv = []
    for line in data:
        numbers = line.split()
        nfloats = map(float,numbers)
        logv.append(nfloats)
        k = k+1
    return logv
    
#############################################################################
# Run the function main on the Swing thread
import sys
class _RunMain(Runnable):
  def __init__(self,main):
    self.main = main
  def run(self):
    self.main(sys.argv)
def run(main):
  SwingUtilities.invokeLater(_RunMain(main))
run(main)


