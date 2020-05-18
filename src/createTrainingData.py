import cv2
import random
import src.util
from src.imageTools import resize, mergeWithBackground, wrangleImages


def yieldMergedImages( backgroundGen, classImagesList, filters, wrangles=list(), 
                     maxImages=3, debugDebug=False, debug=False):
    """@brief merge random 'classImage' 'classImagesList' wrangled using
     one of the wrangeles from into next image taken from
     'backgroundGen'

    sending to the generator something quits geneartion generations,
    merge at most 'maxImages' images (-1 all images found)

    @param backgroundGen generator yielding tuples (imagePath:string,
    image:binary) of background images

    @param classImagesList list of classImages to merge into backgroud
    images taken from 'backgroundGen'
    
    @yield
    dict{classInfo:dict(value:string,type:string,startOrEnd:string),
    classPos:(h,w), classShape:(h,w), mergedImg:binary,
    boundingBox:dict( ymin:int, xmin:int, ymax:int, xmax:int),
    mergeStatus: boolean},

    """
    
    def nextBackgroundImage():
        """@brief Return next image from  'backgroundGen'

        @return tuple (imagePath:string, image:binary)

        """
        try:
            imagePath = next(backgroundGen)
        except StopIteration:
            return( None, None)
        return( (imagePath, cv2.imread(imagePath)) )
    
    def mergePosition( img, mask):
        """@brief Return 'mask' position in 'img' such that mask fits within
        'img' boundaries

        @return random tuple (height:int,width:int) in range
        (maskH:imgH-maskH),(maskW:imgW-maskW)

        """
        iH,iW = img.shape[:2]
        mH,mW = mask.shape[:2]
        return( random.randrange(0,iH-mH), random.randrange(0,iW-mW)  )
    
    def randomResizeWidth( maxWidth=100, minWidthPercent=0.1, minWidthLimit=30):
        """@brief return randomized width for image to merge into backgroud

        @param minWidthLimit min value of width

        @return random value in the range (minWidth,maxWidth), where
        minWidth = max( minWidthLimit, minWidthPercent*maxWidth)

        """
        minWidth= round(minWidthPercent*maxWidth)
        minWidth = max( minWidth, minWidthLimit )
        return( random.randrange(minWidth, maxWidth))
    
    cnt = 0
    while True:
        # quit loop?
        if maxImages > 0 and cnt > maxImages:
            break
        cnt +=1
        
        # find next background image for merge
        backgroundImagePath, img = nextBackgroundImage()
        if backgroundImagePath is None: break
        
        imgH,imgW = img.shape[:2]
        if debugDebug: src.util.showImage( img, "nextBackgroundImage" )

        # find classImage for merge
        # indx = random.choice(range(len(classImagesList)))
        indx = random.randrange(0, len(classImagesList))
        classImage = classImagesList[indx]
        if debug: print( "classImage['imagePath']=", classImage['imagePath'] )
        
        # randomly resize classImage (=cropped image and mask) 
        resizeWidth = randomResizeWidth(maxWidth=imgW//5)
        if debug: print( "resizeWidth", resizeWidth )
        croppedResized = resize( classImage["cropped"], resizeWidth)
        maskResized = resize( classImage["mask"], resizeWidth)
        if debugDebug: src.util.showImage( croppedResized, "croppedResized" )
        # bounding box of classImage
        img2H,img2W = maskResized.shape[:2]
        boundingBox = { "ymin":0, "xmin": 0, "ymax": img2H, "xmax": img2W }

        # randomly wrangle of classImage  (=cropped and mask images)
        croppedResized, maskResized = wrangleImages( croppedResized,
                                                        maskResized, wrangles, filters)
        ## src.util.showImage( croppedResized, "croppedResized" )        
        
        # Find random position where to merge into
        mergePos = mergePosition( img, maskResized )
        if debug: print( "mergePos", mergePos, ", img.shape=", img.shape )

        # Merge resized and wrangeld classImage to background (copy)
        mergedImg, boundingBox = mergeWithBackground( img.copy(), croppedResized, maskResized, mergePos )
        mergeStatus = True if mergedImg is not None else False
        if debug:
            if mergedImg is not None:
                src.util.showImage( mergedImg, "mergedImg" )
            else:
                print( "Could not merge into ", backgroundImagePath )



        # Yield result and check if quitting
        retVal = {
            "classInfo": classImage["classInfo"]
            , "boundingBox" : boundingBox
            , "classPos": mergePos
            , "classShape": maskResized.shape
            , "mergedImg": mergedImg
            , "mergedImgShape":  mergedImg.shape[:2]
            , "status": mergeStatus
        }
        goon = yield retVal
        if goon is not None: break

    

def createTestImage( mergedImage, indx, testSet, imgType="jpg", wrangles=list() ):
    """@brief Finalize 'mergedImage' to 'testImage' using set of
    'wrangeles' 

    @param indx unique test image number

    @param testSet folder name of the test set

     @param mergedImage
    dict{classInfo:dict(value:string,type:string,startOrEnd:string),
    classPos:(h,w), classShape:(h,w), mergedImg:binary, mergeStatus:
    boolean, classname:string}

    @return testImage as 'mergedImage' + dict{testImage:binary,
    imageFilename: string, labelFilename: string, indx:int,
    testSet:string, vocXML: string}

    """
    mergedImage["testImage"], mask = wrangleImages(
        mergedImage['mergedImg'], None, wrangles, None )

    # Generate unique name
    mergedImage["indx"] = indx
    mergedImage["testSet"] = testSet
    ## mergedImage["classname"] = "{0}-{2}".format(mergedImage["classInfo"]["value"], mergedImage["classInfo"]["type"], mergedImage["classInfo"]["startOrEnd"])
    
    fileTrunk = "TSR-image{0:05d}".format(mergedImage["indx"] )
    mergedImage["imageFilename"] = "{0}.{1}".format(fileTrunk, imgType )
    mergedImage["labelFilename"] = "{0}.{1}".format(fileTrunk, "xml" )
    mergedImage["vocXML"] = vocXML( mergedImage )
    
    return( mergedImage, None )
    ##src.util.showImage( img2, "Wrangled " + str(testImage["classInfo"]))

def vocXML( testImage ):
    """"@brief Create xml-string for 'testImage'
    
    @param testImage dict( )
    """
    databaseName="TSR training data"
    return ( f"""<annotation>
	<folder>{testImage["testSet"]}</folder>
	<filename>{testImage["imageFilename"]}</filename>
	<source>
		<database>{databaseName}</database>
		<annotation>classInfo: {testImage["classInfo"]}</annotation>
		<image>flickr</image>
	</source>
	<size>
		<width>{testImage["mergedImgShape"][1]}</width>
		<height>{testImage["mergedImgShape"][0]}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>{testImage["classname"]}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{testImage['boundingBox']["xmin"]}</xmin>
			<ymin>{testImage['boundingBox']["ymin"]}</ymin>
			<xmax>{testImage['boundingBox']["xmax"]}</xmax>
			<ymax>{testImage['boundingBox']["ymax"]}</ymax>
		</bndbox>
	</object>
</annotation>""")

