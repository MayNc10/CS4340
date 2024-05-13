import deeplake
import numpy as np
from numpy import savez_compressed
path_prepend = "Final Project/"

ds = deeplake.load("hub://activeloop/kuzushiji-kanji")
imgs = np.zeros((64,64), np.uint8).reshape(1,64,64) #Dummy black image to setup the ndarray

print(len(ds))

for idx in range(len(ds)):
    image = ds['images'][idx].data()['value']
    #print(image)
    imgs = np.concatenate([imgs, image.reshape(1, 64, 64)])
    print(f"idx: {idx}")

labels = ds['labels'].data()['value']
savez_compressed(path_prepend + 'dataset/kkanji-imgs.npz', imgs)
savez_compressed(path_prepend + 'dataset/kkanji-labels.npz', labels)


