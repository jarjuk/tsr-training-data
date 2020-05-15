import os
import re
import cv2
import numpy as np

from  src.util import showImage

def iterateClassImages(imagesDirectory):
    """@return tuple (classinfo:dic(value:, type:,
    startOrEnd:),imagePath:string) for image files in
    'imagesDirectory'

    """
    validImageFile = re.compile(".+\.(png|jpg|jpeg)")

    def extractClassInfo(filePath):
        fileTrunck = os.path.splitext(os.path.basename(filePath))[0]
        return (dict(
            zip(["value", "type", "startOrEnd"],
                os.path.basename(fileTrunck).split("-"))))

    return ((extractClassInfo(imageDirEntry.path), imageDirEntry.path)
            for imageDirEntry in filter(
                lambda e: re.search(validImageFile, e.name),
                os.scandir(imagesDirectory)))


def maskImage( img, debug = False, debugDebug = False ):
    """Extract largest image contour in 'img' and use it create mask to
    crop image from background.

    @return tuple (out, mask)

    """
    ## Convert to gray-scale
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # if debugDebug: showImage(gray, "Gray")

    ## Find edges
    (threshold1,
     threshold2) = (30, 100)  # most of countour are there, maybe too many?
    thld = cv2.Canny(gray, threshold1, threshold2)
    if debugDebug: showImage(thld, "Edges")

    # make edges stronger using 'dilate'
    kernel = np.ones((3,3),np.uint8)
    thld = cv2.dilate(thld,kernel,iterations=1)
    if debugDebug: showImage(thld, "Dilated edegs")


    ## Find countours i.e. curves joining all the continuous points
    ## (along the boundary), having same color or intensity
    contours, _ = cv2.findContours(thld.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if debugDebug:
        showImage(cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 1),
                  "contours")

    # "crop" (actually mask out) using largest contour area
    # https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour
    contours = sorted(contours, key = cv2.contourArea, reverse = True)    
    mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
    out = np.zeros_like(mask)  # Extract out the object and place into output image
    cv2.drawContours(mask, contours, 0, (255,255,255), -1) # Fill largest contour (indx 0) in mask
    out[mask == (255,255,255)] = img[mask == (255,255,255)]

    return( out, mask)
    

def cropClassImage(classImageTuple, debug=False, debugDebug=False):
    """Extract largest image contour of the image in path
     classImageTuple[1], and use it create mask to crop image from
     background.

    @param classImageTuple (classInfo,imagePath)

    @return: dict(imagePath:, status:, classInfo:, img:, cropped:,
    mask:)

    """
    (classInfo, imagePath) = classImageTuple
    
    if debugDebug:
        print("classInfo={0}, imagePath={1}".format(classInfo, imagePath))

    ## Read original image
    img = cv2.imread(imagePath)
    # if debugDebug: showImage(img)
    #(h, w, ch) = img.shape
    #img_area = float(w * h)

    ## Convert to gray-scale
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # if debugDebug: showImage(gray, "Gray")

    ## Find edges
    (threshold1,
     threshold2) = (30, 100)  # most of countour are there, maybe too many?
    thld = cv2.Canny(gray, threshold1, threshold2)
    if debugDebug: showImage(thld, "Edges")

    # make edges stronger using 'dilate'
    kernel = np.ones((3,3),np.uint8)
    thld = cv2.dilate(thld,kernel,iterations=1)
    if debugDebug: showImage(thld, "Dilated edegs")


    ## Find countours i.e. curves joining all the continuous points
    ## (along the boundary), having same color or intensity
    contours, _ = cv2.findContours(thld.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if debugDebug:
        showImage(cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 1),
                  "contours")

    # "crop" (actually mask out) using largest contour area
    # https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour
    contours = sorted(contours, key = cv2.contourArea, reverse = True)    
    mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
    out = np.zeros_like(mask)  # Extract out the object and place into output image
    cv2.drawContours(mask, contours, 0, (255,255,255), -1) # Fill largest contour (indx 0) in mask
    out[mask == (255,255,255)] = img[mask == (255,255,255)]

    result = {
        "imagePath": imagePath
        , "status": True
        , "classInfo": classInfo
        , "gray": gray
        , "img": img
        , "contours": cv2.drawContours(img.copy(), contours, 0, (0, 255, 0), 1)
        , "cropped": out
        , "mask": mask
    }
    return(result)

def yieldCroppedClassImages(imageDirectory):
    """@yield classImage:dict(imagePath:, status:, classInfo:, img:, cropped:, mask:)
    for all images in 'imageDirectory'

    """
    for x in iterateClassImages(imageDirectory):
        yield cropClassImage(x, debug=False, debugDebug=False)          

