import os.path
import sys
from absl import app, flags, logging
from absl.flags import FLAGS
import random
import cv2

import src.createTrainingData
import src.backgrounds
import src.classImages

# control
flags.DEFINE_integer('debug', -1, '-3=fatal, -1=warning, 0=info, 1=debug')
flags.DEFINE_integer('seed', None, 'Random seed (default None)')
flags.DEFINE_boolean('interactive', False, 'show training data')
flags.DEFINE_integer("maxImages", 7, "Number of images to generate")

# input
flags.DEFINE_string('classImages', "Images/signs", "path to class images")
flags.DEFINE_string('backgrounds', 'backgrounds', 'path to backgroud images')
flags.DEFINE_string('testSet', "TSR1", 'Test set name (=folder in output -directory)')

# class image filters
flags.DEFINE_multi_integer( "rotates", [10,4  ,-5, -10], "Rotate filters used on classImages")
flags.DEFINE_multi_integer( "blur", [3,5], "Blur filters used on classImages")
flags.DEFINE_multi_integer( "brightness", [-70,-20], "Brigness filters used on classImages")
flags.DEFINE_multi_string( "perspective", [ "45,-10", "10,10", "-30,10" ], "Possible perspective values")
flags.DEFINE_multi_string( "classImageWrangles", [ "blur", "rotate", "brightness", "perspective" ], "Filter wrangles to run on class images" )

# test image filters
flags.DEFINE_multi_integer( "brightness2", [-30,-20], "Brigness filters used on test images")
flags.DEFINE_multi_integer( "blur2", [1,2,3], "Blur filters used on test images")
flags.DEFINE_multi_string( "testImageWrangles", [ "brightness", "blur" ], "Filter wrangles to run on test images (=final result)" )

# output
flags.DEFINE_string('images', "out/tsrVOC/JPEGImages", 'output folder where test images are written')
flags.DEFINE_string( 'annotations', "out/tsrVOC/Annotations" , "path to annotations folder")
flags.DEFINE_string('classes', "out/tsrVOC/classes.txt", 'path file where classnames are written')
flags.DEFINE_string( "classesTemplate",
                     "{value}-{startOrEnd}", "template to interpolate classes file lines")

flags.DEFINE_string('imagelist', "out/tsrVOC/imagelist.txt",
                    'path file where list of images are written')
flags.DEFINE_string( "imagelistTemplate",
                     "{imageFileBasename} -1\n", "template to interpolate imagelist file lines")


## flags.DEFINE_list( "rotates", "-32,-10,-5,3,35,45", "rotate angles to wrangle class images")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_filedir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)        

def main(_argv):


    def progress(count, total, status=''):
        # https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stderr.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stderr.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-c

    logging.set_verbosity(FLAGS.debug)
    random.seed( FLAGS.seed )
    logging.info( "Random seed {0}".format(FLAGS.seed))

    # Default values
    if FLAGS.annotations is None:
        FLAGS.annotations = os.path.join( FLAGS.images, "annotations")

    
    logging.info('starting')

    def resetBackgroundGen():
        return( src.backgrounds.yieldBackgroundImagePaths(FLAGS.backgrounds) )

    # Generator yielding backgroud images
    backgroundGen = resetBackgroundGen()

    # Paste random classImages to back
    classImagesList = list( src.classImages.yieldCroppedClassImages(FLAGS.classImages) )

    # Set of filters randomly maninpulating classImage before pasting
    # into background
    classImagesFilters = src.imageTools.Filters(
        { "rotate" : FLAGS.rotates, "blur": FLAGS.blur
          , "brightness": FLAGS.brightness
          , "perspective": FLAGS.perspective
    })
    classImageWrangles = classImagesFilters.imageWrangles( wrangleTypes = FLAGS.classImageWrangles)
    logging.info( "classImageWrangles={0}".format( classImageWrangles))

    # Wranlers to run on test images (=finalize result)

    testImagesFilters = src.imageTools.Filters(
        { 
            "brightness": FLAGS.brightness2
            , "blur": FLAGS.blur2
        })    
    testImageWrangles = testImagesFilters.imageWrangles(wrangleTypes = FLAGS.testImageWrangles)
    logging.info( "testImageWrangles={0}".format( testImageWrangles))    
    
    

    logging.info( "Merge class images '{0}' with backgrounds '{1}' to a test set '{2}' with image  '{3}' and annotations {4} folders ".format(
        FLAGS.classImages, FLAGS.backgrounds, FLAGS.testSet, FLAGS.images, FLAGS.annotations ))

    # truncate imagelist file
    imagesDirectory = os.path.join(FLAGS.images)
    annotationDirectory =  os.path.join(FLAGS.annotations)

    # make sure that directories exist 
    ensure_dir( imagesDirectory )
    ensure_dir( annotationDirectory )
    ensure_filedir( FLAGS.imagelist )
    ensure_filedir( FLAGS.classes )    


    # empty imagelist
    open( FLAGS.imagelist, "w").close()
    
    # Collect unique class names
    classnames = set()
    imageCount = 0
    while True:
        
        for mergedImage in  src.createTrainingData.yieldMergedImages(
                    backgroundGen,
                    classImagesList, classImagesFilters,
                    wrangles=classImageWrangles,
                    debug=False, debugDebug=False ):

            if imageCount >= FLAGS.maxImages:
                break


            # iterate 'backgroundGen', choose random image from
            # 'classImagesList', and run random wrangler from
            # 'classImageWrangles' and merge classImage with background

            logging.info( "classInfo value: {0} type: {1} startOrEnd: {2}".format(
                mergedImage["classInfo"]["value"], mergedImage["classInfo"]["type"], mergedImage["classInfo"]["startOrEnd"], mergedImage["classInfo"]) )

            # Use template to define class name
            mergedImage["classname"] = FLAGS.classesTemplate.format(**mergedImage["classInfo"] )

            # actual merge
            testImage, _ = src.createTrainingData.createTestImage(
                mergedImage, imageCount , testSet=FLAGS.testSet,
                wrangles=testImageWrangles, filters=testImagesFilters )

            if FLAGS.interactive: 
                imgBox = src.imageTools.drawBoundingBox( testImage["testImage"], testImage["boundingBox"] )
                src.util.showImage( imgBox, "Test image " + str(testImage["classInfo"] ) )



            # Collect classnames to a set 

            classnames.add(testImage["classname"])

            # write 1) image, 2) label xml,  3) imagelist file entry
            imageCount = imageCount + 1
            cv2.imwrite( os.path.join( imagesDirectory, testImage["imageFilename"]), testImage["testImage"] )
            with open( os.path.join( annotationDirectory, testImage["labelFilename"]), "w+" ) as labelFile:
                labelFile.write( testImage["vocXML"] )
            with open( FLAGS.imagelist, "a") as imagelistFile:
                ## imagelistFile.write( FLAGS.imagelistTemplate.format( ).format( )testImage["imageFilename"] + -1"\n" )
                testImage["imageFileBasename"] = os.path.splitext( os.path.basename( testImage["imageFilename"] ))[0]
                imagelistFileRow = FLAGS.imagelistTemplate.format(**testImage )
                imagelistFile.write( str(imagelistFileRow)  )

            progress( imageCount, FLAGS.maxImages, "Images")

        # all requestesd images done
        if imageCount >= FLAGS.maxImages:
            break
        else:
            # Restart generator yielding backgroud images
            backgroundGen = resetBackgroundGen()



    # create classes file with unique classnames
    classNameLines = [ cName+"\n" for cName in sorted(classnames) ]
    with open( FLAGS.classes, "w+" ) as classFile:
        classFile.writelines( classNameLines )
            
    print( "Created {0} test images in '{1}' with annotations in '{2}' folders".format(
        imageCount, FLAGS.images,  FLAGS.annotations ))
    print( " class names in '{0}' and list of images '{1}'".format(
        FLAGS.classes, FLAGS.imagelist))
    print( " input from from class images '{0}' and backgrounds'{1}' folders".format(
        FLAGS.classImages, FLAGS.backgrounds ))



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    
