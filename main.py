import numpy as np
import cv2
import matplotlib.pyplot as plt
from build import libdepthComp

disp = cv2.imread("/mnt/md0/Data/CityScapes/disparity/val/munster/munster_000121_000019_disparity.png", cv2.IMREAD_UNCHANGED)
labelIn = cv2.imread("/mnt/md0/Data/CityScapes/gtFine/val/munster/munster_000121_000019_gtFine_labelTrainIds.png")

labelIn[labelIn == 255] = 19
labelIn = labelIn.astype(np.float32)

fx = 2256.47
baseline = 0.209313

disp_cp = np.copy(disp)
disp, _ = cv2.filterSpeckles(disp.astype(np.int16), 0, 200, 100)

def disp_to_depth(disp):
    mask = disp != 0
    disp = mask * ((disp.astype(np.float32) - 1.) / 256.)
    with np.errstate(divide='ignore'):
        depth = (baseline * fx) / disp
    depth[disp == 0] = 0
    return depth

depth = disp_to_depth(disp)

depthOut = libdepthComp.depthComplete(disp.astype(np.float32), labelIn.astype(np.float32))

plt.subplot(2,1,1)
plt.imshow(depth)
plt.subplot(2,1,2)
plt.imshow(disp_to_depth(depthOut))
plt.show()