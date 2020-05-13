import cv2
import numpy as np

class BreakIt( Exception ):
    pass

def showImage(img, header="Image"):
    img2 = np.array(img)
    cv2.imshow(header, img2)
    key = cv2.waitKey(0)
    ## print("key=", key)
    cv2.destroyAllWindows()
    ## quit on esc
    if (key == 27): raise BreakIt

