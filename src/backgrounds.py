import os
import re

def yieldBackgroundImagePaths( rootdir ):
    """@return generator to recurse 'rootdir' and yield path to
    (png,jpg, jpeg) imagefiles """
    validImageFile = re.compile(".+\.(png|jpg|jpeg|JPG|JPEG|PNG)")
    for root, subdir, files in os.walk(rootdir):
        for file in files:
            if re.search(validImageFile, file ):
                yield os.path.join( root, file )



