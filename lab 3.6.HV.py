from rasterio.plot import show
import rasterio
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
import scipy
from keras.preprocessing.image import img_to_array

image = rasterio.open("C:\\Users\\ma\\project\\Scripts\\data\\lab3\\data\\HVshahdadpur.tif")
data = image.read()
data = data.transpose(1, 2, 0)


samples = expand_dims(data, 0)
datagen = ImageDataGenerator(zoom_range=[ 1, 8])
it = datagen.flow(samples, batch_size=1)


plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint16')
    plt.imshow(result)

plt.show()