from PIL import Image
import numpy as np
import os
def maskConvert(mask_path):
    mask=Image.open(mask_path)
    if mask.mode=='RGB':
        mask=mask.convert('L')
        print(mask_path)
        mask.save(mask_path)
if __name__=='__main__':
    #masks_path='../data/BUSI-256/masks/'
    masks_path='../data/isic2018/train/masks/'
    images=os.listdir(masks_path)
    for img in images:
        maskConvert(masks_path+img)