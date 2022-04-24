
import cv2
import os
import pydicom

pneumo_inputdir = r'PNEUMOTHORAX/'
normal_inputdir=r'NORMAL/'
pneumo_outdir = r'PNG/'
#os.mkdir(outdir)

pneumo_list = [ f for f in  os.listdir(pneumo_inputdir)]
normal_list = [i for i in os.listdir(normal_inputdir)]

for f in pneumo_list:
    ds = pydicom.read_file(pneumo_inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    cv2.imwrite(pneumo_outdir + f.replace('.dcm','.png'),img)



