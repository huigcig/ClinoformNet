"""
Jython utilities for seismic stratigraphic interpretation.
Author: Xinming Wu, Unversity of Science and Technology of China
Version: 2020.12.31
"""
import math
from common import *

#############################################################################
# Internal constants

_datdir = "/media/xinwu/disk-3/cnooc-zj/data/"
_datdir = "../../../../git/cnooc-zj/data/"
_datdir = "../../data/clino-test/model"
_F3fsmdir = "../../data/clinoformdata/model1"
_F3dir = "../../data/F3_block_noise"
_F3clinodir = "../../data/F3d"
_XMclinodir = "../../data/"
_clinodir = "../../data/New_clino/model"
#############################################################################
# Setup
# Setup
def setupForSubset(name):
  """
  Setup for a specified directory includes:
    seismic directory
    samplings s1,s2,s3
  Example: setupForSubset("hongliu")
  """
  global seismicDir
  global clinovolDir
  global welllogDir
  global s1,s2,s3
  global n1,n2,n3
  global sz,sl,sc
  global nz,nl,nc
  if name=="clino-loop":
    seismicDir = "../../data/New_clino/model"
    
  elif name=="models":
    seismicDir = "../../data/models/model"
#     n1,n2,n3 = 561,700,204
#     s1,s2,s3 = Sampling(n1),Sampling(n2),Sampling(n3)
  elif name=="classic_models":
    seismicDir = "../../data/classic_models/model"  
  else:
    print "unrecognized subset:",name
    System.exit

def getSamplings():
  return s1,s2,s3

def getSeismicDir():
  return seismicDir

#############################################################################
# read/write images
def readImageclinoL(name,ip,nc1,nc2,nc3):
  fileName = seismicDir+str(ip)+"/"+name+".dat"
  image = zerofloat(nc1,nc2,nc3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImageclinoL2d(name,ip,nc1,nc2):
  fileName = seismicDir+str(ip)+"/"+name+".dat"
  image = zerofloat(nc1,nc2)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImagefieldL2d(name,ip,nf1,nf2):
   fileName = name + str(ip) +".dat"
   image = zerofloat(nf1,nf2)
   ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
   ais.readFloats(image)
   ais.close()
   return image

def writeImagefieldL2d(name,image,ip):
  fileName = name + str(ip) + ".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  aos.close()
  return image

def writeImageclinoL(name,image,ip):
  fileName = seismicDir+str(ip)+"/"+name+".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  aos.close()
  return image

def writeImageclinoL2d(name,image,ip):
  fileName = seismicDir+str(ip)+"/"+name+".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  aos.close()
  return image

def readImageNoisedtL(name,index):
  """ 
  Reads an image from a file with specified name.
  name: base name of image file; e.g., "tpsz"
  """
  NoiseDir = "../../data/noise/"
  fileName = NoiseDir+name+".dat" 

  if index < 2:
    ns1,ns2,ns3 = 400,1902,461 # cut clino 2ms noise data
  elif index >= 2 and index < 4:
    ns1,ns2,ns3 = 400,2000,288 # cut clino 2ms noise data
  elif index >= 4  and index < 6: 
    ns1,ns2,ns3 = 800,4400,2 # cut clino 2ms noise data
  else:
    print "wrong"
  image = zerofloat(ns1,ns2,ns3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImageNoiseL(name):
  """ 
  Reads an image from a file with specified name.
  name: base name of image file; e.g., "tpsz"
  """
  NoiseDir = "../../data/F3_block_noise/"
  ns1,ns2,ns3 = 240,951,591 # cut clino 3ms noise data
  fileName = NoiseDir+name+".dat"
  image = zerofloat(ns1,ns2,ns3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImageL(name):
  """ 
  Reads an image from a file with specified name.
  name: base name of image file; e.g., "tpsz"
  """
  fileName = seismicDir+name+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImage(name):
  """ 
  Reads an image from a file with specified name.
  name: base name of image file; e.g., "tpsz"
  """
  fileName = seismicDir+name+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def readImage2(m1,m2,name):
  fileName = seismicDir+name+".dat"
  image = zerofloat(m1,m2)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image
  
def readSlice3(name):
  fileName = seismicDir+name+".dat"
  n1,n2 = s1.count,s2.count
  image = zerofloat(n1,n2)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def writeImageL(name,image):
  """ 
  Writes an image to a file with specified name.
  name: base name of image file; e.g., "tpgp"
  image: the image
  """
  fileName = seismicDir+name+".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  aos.close()
  return image

def writeImage(name,image):
  """ 
  Writes an image to a file with specified name.
  name: base name of image file; e.g., "tpgp"
  image: the image
  """
  fileName = seismicDir+name+".dat"
  aos = ArrayOutputStream(fileName)
  aos.writeFloats(image)
  aos.close()
  return image

#############################################################################
# read/write fault skins

def skinName(basename,index):
  return basename+("%05i"%(index))
def skinIndex(basename,fileName):
  assert fileName.startswith(basename)
  i = len(basename)
  return int(fileName[i:i+5])

def listAllSkinFiles(basename):
  """ Lists all skins with specified basename, sorted by index. """
  fileNames = []
  for fileName in File(seismicDir).list():
    if fileName.startswith(basename):
      fileNames.append(fileName)
  fileNames.sort()
  return fileNames

def removeAllSkinFiles(basename):
  """ Removes all skins with specified basename. """
  fileNames = listAllSkinFiles(basename)
  for fileName in fileNames:
    File(seismicDir+fileName).delete()

def readSkin(basename,index):
  """ Reads one skin with specified basename and index. """
  return FaultSkin.readFromFile(seismicDir+skinName(basename,index)+".dat")

def readSkins(basename):
  """ Reads all skins with specified basename. """
  fileNames = []
  for fileName in File(seismicDir).list():
    if fileName.startswith(basename):
      fileNames.append(fileName)
  fileNames.sort()
  skins = []
  for iskin,fileName in enumerate(fileNames):
    index = skinIndex(basename,fileName)
    skin = readSkin(basename,index)
    skins.append(skin)
  return skins

def writeSkin(basename,index,skin):
  """ Writes one skin with specified basename and index. """
  FaultSkin.writeToFile(seismicDir+skinName(basename,index)+".dat",skin)

def writeSkins(basename,skins):
  """ Writes all skins with specified basename. """
  for index,skin in enumerate(skins):
    writeSkin(basename,index,skin)

from org.python.util import PythonObjectInputStream
def readObject(name):
  fis = FileInputStream(seismicDir+name+".dat")
  ois = PythonObjectInputStream(fis)
  obj = ois.readObject()
  ois.close()
  return obj
def writeObject(name,obj):
  fos = FileOutputStream(seismicDir+name+".dat")
  oos = ObjectOutputStream(fos)
  oos.writeObject(obj)
  oos.close()
    
