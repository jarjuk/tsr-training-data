import cv2
import random
import numpy as np
import imutils
import math

import functools

from absl import logging

def boundingBox( img ):
   """@return  dict( ymin:int, ymax:int, xmin: int, xman: int)
   """
   h, w = img.shape[:2] # image shape has 3 dimensions
   boundingBox = { "ymin": 0, "xmin":0, "ymax": h, "xmax": w }
   return( boundingBox)

def boundingBox2points( boundingBox ):
    """@return array[ 4x array[2]]  """
    return( np.float32( [ [boundingBox["xmin"], boundingBox["ymin"]],
                         [boundingBox["xmax"], boundingBox["ymin"]],
                         [boundingBox["xmax"], boundingBox["ymax"]],
                         [boundingBox["xmin"], boundingBox["ymax"]]
   ]))



def resize( img, width):
    """@return resized 'img' to 'width', aspect ratio kept"""
    img = imutils.resize( img, width=width)
    return( img )

def resize_image( img, width ):
    """@return resized 'img' to 'width', aspect ratio kept"""
    img = imutils.resize( img, width=width)
    return(  imutils.resize( img, width=width) )

def multi_filter( img, funcs ):
    return( functools.reduce( lambda img, f: f(img), funcs, img ) )


def blur_image( img, kernelSize=5):
    """@return blulled 'img'"""
    dst = cv2.blur(img,(kernelSize,kernelSize))
    return( dst )
    

def perspective_image( img, rots):
    """
    @param rots tuple(int, int)
    """
    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(theta):
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R
    
    def findDsize( bb, M ):
        """@return dsize, offset where 'dsize': (width:int, height:int) of
           transformed image, offset: (x:int, y:int)

        """
        xmin = ymin = xmax = ymax = 0
        for corner in boundingBox2points( bb ):
            # print( "corner.shape", corner.shape, ", corner=", corner  )
            p = [corner[0], corner[1], 0]
            # map corner point
            newP = np.dot( M, np.array(p))
            # print( "corner.shape", corner.shape, ", corner=", corner, "newP", newP  )
            xmin = min( xmin, newP[0])            
            ymin = min( ymin, newP[1])            
            xmax = max( xmax, newP[0])            
            ymax = max( ymax, newP[1])
            
        dsize = (int(xmax-xmin), int(ymax-ymin))
        offset = (int(min(0,xmin)), int(min(0,ymin)))
        # print( "dsize=", dsize )
        return( dsize , offset )

    
    def offsetT( offset):
        """@return 3x3 matrix tranformin image offset (x,y)"""
        M = np.eye(3)
        M[0,2] = -offset[0]
        M[1,2] = -offset[1]
        return( M )

    # add possible rotation
    if len(rots) == 2: rots = rots +(0,)
    
    # maps rots --> 2-Dtransformation matrix
    rads =  [math.radians(r) for r in rots ]
    M = eulerAnglesToRotationMatrix( rads )
    # 2D transformation (ignore Z)
    M[2] = [0,0,1]

    # find target image size and offset of the image
    dsize, offset = findDsize( boundingBox(img), M )

    # translation matrix
    T = offsetT( offset )

    # perspective transformation 'M' and offset translation 'T'
    dst = cv2.warpPerspective(img,np.dot(T, M),dsize)
    return( dst )



def brightness_image( img, adjust):
    """@return image brightness adjusted by adjustBrightness"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.add(hsv[:,:,2], adjustBrightness, dst=hsv[:,:,2])
    v =cv2.add(hsv[:,:,2], adjust)
    hsv[:,:,2] = v
    img  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return( img )


def gamma_image(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return( cv2.LUT(image, table) )

# https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides/47248339
# rotating images while avoiding cropping the image.

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return( rotated_mat )

# from  scipy import ndimage

# def rotate_image2(img, angle):
#     return( ndimage.rotate(img, angle))



def mergeWithBackground( background, cropped, mask, pos=(0,0) ):
    """Write 'cropped' image to 'background' to position 'pos' (height,
    width) using 'mask'

    @return merged image and boundingBox (img:binary,
    boundingBox:dict(ymin:int, ymax=int, xmin:int, xmax:in))

    """

    ## Extract and validate dimensions
    oH,oW = pos                   # offset 
    mH, mW = mask.shape[:2]       # mask
    bH, bW = background.shape[:2] # background
    if pos[0]<0 or pos[1] < 0 or (oH+mH) > bH or (oW+mW)>bW:
        return( None)

    ## Write cropped to background offset using mask
    masked=mask==255
    roi = background[oH:oH+mH,oW:oW+mW]
    roi[np.where(masked)] = cropped[np.where(masked)]
    logging.debug( "mergeWithBackground, before: background.shape {0}  roi.shape={1}".format( background.shape, roi.shape) )
    boundingBox = { "ymin": oH, "xmin": oW, "ymax": oH+mH, "xmax": oW+mW }
    #background[oH:oH+mH,oW:oW+mW] = roi
    background[boundingBox["ymin"]:boundingBox["ymax"], boundingBox["xmin"]:boundingBox["xmax"]] = roi
    logging.debug( "mergeWithBackground, after: background.shape {0}  roi.shape={1}".format( background.shape, roi.shape) )    

    
    return(background, boundingBox)



class Filters:

    ROTATES= [-32, -10, -2, 1, 5, 28, 35, 45]
    RESIZES=[50,100,200]
    BLURS=[3,6]
    BRIGHTNESSES=[10,50,100]
    
    def rotates():
        return Filters.ROTATES
    def resizes():
        return Filters.RESIZES
    def blurs():
        return Filters.BLURS
    def brightnesses():
        return Filters.BRIGHTNESSES

    def rotateNames( noop= True ):
        names = [ "rotate{0}".format(k) for  k in Filters.rotates() ]
        if noop:
            names = names + [ "noop"]
        return( names )
    def blurNames( noop= True):
        names = [ "blur{0}".format(k) for k in Filters.blurs() ]
        if noop:
            names = names + [ "noop"]
        return( names )
    def brightnessNames( noop = True ):
        names = [ "brightness{0}".format(k) for k in Filters.brightnesses() ]
        if noop:
            names = names + [ "noop"]
        return( names )

    
simpleWrangles = {
    "noop" : lambda img, boundingBox:  (img, boundingBox)
}

def createLambda(  f, k ):
    """Short cut and create contexted for 'k'"""
    return( lambda img, boundingBox: f( img, boundingBox, k) )

simpleWrangles.update({ "rotate{0}".format(k): createLambda( rotate_image, k ) for k in Filters.rotates() })
simpleWrangles.update({ "resize{0}".format(k): createLambda( resize_image, k ) for k in Filters.resizes() })
simpleWrangles.update({ "blur{0}".format(k): createLambda ( blur_image, k ) for k in Filters.blurs() })
simpleWrangles.update({ "brightness{0}".format(k): createLambda( brightness_image, k ) for k in Filters.brightnesses() })

logging.debug( "simpleWrangles.keys()", simpleWrangles.keys())

    


def wrangleImages( imgs, wrangles, wrangleDict=simpleWrangles ):
    """Wrangle each 'img' in 'imgs' (note that imgs may be a list or a
    single image) using a random 'wrangler' in 'wrangles'

    @param imgs is simple image or list of images

    @param wranles array[string|[string]] (=array of strings or array
    of array of strings)

    @return wrangled 'imgs'

    """

    def runWrangler( img, wrangleDef, wrangleDict ):
        wrangleNames = wrangleDef if isinstance( wrangleDef, list) else [wrangleDef]
        for wrangleName in wrangleNames:
            img = wrangleDict[wrangleName](img )

        return( img )

        
    retVal = imgs
    if  len(wrangles) > 0:
        indx = random.choice(range(len(wrangles)))
        wranglerDef= wrangles[indx]
        # wrangler = wrangleDict[wranglerName]
        if isinstance(imgs, list):
            retVal = []
            for i in range(len(imgs)):
                dst = runWrangler( imgs[i], wranglerDef, wrangleDict )
                # dst = wrangler( imgs[i] )
                retVal.append(dst)
            return( retVal, boundingBox )
        else:
            retVal  = runWrangler( imgs,  wranglerDef, wrangleDict )
            # retVal = wrangler( imgs  )
            return( retVal, boundingBox)
    return( (retVal, boundingBox) )





def drawBoundingBox( img, boundingBox):
    """@param boundingBox dict( ymin:int, xmin:int, ymax:int, xmax:int)"""
    color = (0,255,0)
    return cv2.rectangle( img, (boundingBox["xmin"], boundingBox["ymin"]), (boundingBox["xmax"], boundingBox["ymax"]), color, 2)
