import os
import sys
if (sys.version_info > (3, 0)):
    # Python 3
    import configparser as cp
else:
    # Python 2
    import ConfigParser as cp
 

def get_config():
    file = os.path.join( os.path.dirname( os.path.realpath( __file__ ) ), 'config.ini' )  
    parser = cp.ConfigParser()
    parser.read( file )
    return parser

