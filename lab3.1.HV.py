from rasterio.plot import show
import geos
import tensorflow
import rasterio
import geotiff
import tifffile
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image


image = rasterio.open("C:\\Users\\ma\\project\\Scripts\\data\\lab3\\data\\HVshahdadpur.tif")
show(image)