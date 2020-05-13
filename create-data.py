import itertools
import os.path
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2

import src.createTrainingData
import src.backgrounds
import src.classImages

# control
flags.DEFINE_integer('debug', -1, '-3=fatal, -1=warning, 0=info, 1=debug')
flags.DEFINE_boolean('interactive', False, 'show training data')
flags.DEFINE_integer("maxImages", -1, "Number of images to generate, -1=proces all background images")
flags.DEFINE_string( "imagelistTemplate",
                     "{imageFileBasename} -1\n", "template to interpolate imagelist file lines")
flags.DEFINE_string( "classesTemplate",
                     "{value}-{startOrEnd}", "template to interpolate classes file lines")

# input
flags.DEFINE_string('classImages', "Images/signs", "path to class images")
flags.DEFINE_string('backgrounds', 'backgrounds', 'path to backgroud images')
flags.DEFINE_string('testSet', "TSR1", 'Test set name (=folder in output -directory)')

# output
flags.DEFINE_string('images', "out/tsrVOC/JPEGImages", 'output folder where test images are written')
flags.DEFINE_string(
    'annotations', "out/tsrVOC/Annotations"
    , "path to  annotations folder (defaults 'annotations' under images folder)")
flags.DEFINE_string('classes', "out/tsrVOC/classes.txt", 'path file where classnames are written')
flags.DEFINE_string('imagelist', "out/tsrVOC/imagelist.txt",
                    'path file where list of images are written')


## flags.DEFINE_list( "rotates", "-32,-10,-5,3,35,45", "rotate angles to wrangle class images")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_filedir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)        

def main(_argv):

    logging.set_verbosity(FLAGS.debug)

    # Default values
    if FLAGS.annotations is None:
        FLAGS.annotations = os.path.join( FLAGS.images, "annotations")

    
    logging.info('starting')

    # Generator yielding backgroud images
    backgroundGen = src.backgrounds.yieldBackgroundImagePaths(FLAGS.backgrounds)

    # Paste random classImages to back
    classImagesList = list( src.classImages.yieldCroppedClassImages(FLAGS.classImages) )

    # Set of filters randomly maninpulating classImage before background past
    classImageWrangles = [ list(w) for w in itertools.product(
        src.imageTools.Filters.blurNames(),
        src.imageTools.Filters.brightnessNames(),
        src.imageTools.Filters.rotateNames() )]
    logging.info( "classImageWrangles={0}".format( classImageWrangles))

    
    # First image number
    indx = 1

    logging.info( "Merge class images '{0}' with backgrounds '{1}' to a test set '{2}' with image  '{3}' and annotations {4} folders ".format(
        FLAGS.classImages, FLAGS.backgrounds, FLAGS.testSet, FLAGS.images, FLAGS.annotations ))

    # truncate imagelist file
    imagesDirectory = os.path.join(FLAGS.images)
    annotationDirectory =  os.path.join(FLAGS.annotations)

    # make sure that directoies exist 
    ensure_dir( imagesDirectory )
    ensure_dir( annotationDirectory )
    ensure_filedir( FLAGS.imagelist )
    ensure_filedir( FLAGS.classes )    

    
    open( FLAGS.imagelist, "w").close()
    
    # Collect unique class names
    classnames = set()
    for indx, mergedImage in enumerate(src.createTrainingData.yieldMergedImages(
            backgroundGen, classImagesList, maxImages=FLAGS.maxImages, wrangles=classImageWrangles,
            debug=False, debugDebug=False )):

        # iterate 'backgroundGen', choose random image from
        # 'classImagesList', and run random wrangler from
        # 'classImageWrangles' and merge classImage with background

        logging.info( "value: {0}".format(mergedImage["classInfo"]["value"], mergedImage["classInfo"]["type"], mergedImage["classInfo"]["startOrEnd"], mergedImage["classInfo"]) )

        # Use template to define class name
        mergedImage["classname"] = FLAGS.classesTemplate.format(**mergedImage["classInfo"] )

        testImage, _ = src.createTrainingData.createTestImage( mergedImage, indx , testSet=FLAGS.testSet  )

        if FLAGS.interactive: 
            imgBox = src.imageTools.drawBoundingBox( testImage["testImage"], testImage["boundingBox"] )
            src.util.showImage( imgBox, "Test image " + str(testImage["classInfo"] ) )



        # Collect classnames to a set 
       
        classnames.add(testImage["classname"])

        # write 1) image, 2) label xml,  3) imagelist file entry
        cv2.imwrite( os.path.join( imagesDirectory, testImage["imageFilename"]), testImage["testImage"] )
        with open( os.path.join( annotationDirectory, testImage["labelFilename"]), "w+" ) as labelFile:
            labelFile.write( testImage["vocXML"] )
        with open( FLAGS.imagelist, "a") as imagelistFile:
            ## imagelistFile.write( FLAGS.imagelistTemplate.format( ).format( )testImage["imageFilename"] + -1"\n" )
            testImage["imageFileBasename"] = os.path.splitext( os.path.basename( testImage["imageFilename"] ))[0]
            imagelistFileRow = FLAGS.imagelistTemplate.format(**testImage )
            imagelistFile.write( str(imagelistFileRow)  )
            

    # create classes file with unique classnames
    classNameLines = [ cName+"\n" for cName in classnames  ]
    with open( FLAGS.classes, "w+" ) as classFile:
        classFile.writelines( classNameLines )
            
    print( "Created {0} test images in '{1}' with annotations in '{2}' folders".format(
        indx+1, FLAGS.images,  FLAGS.annotations ))
    print( " class names in '{0}' and list of images '{1}'".format(
        FLAGS.classes, FLAGS.imagelist))
    print( " input from from class images '{0}' and backgrounds'{1}' folders".format(
        FLAGS.classImages, FLAGS.backgrounds ))



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    
