from .. import config 
import os
import sys
import imageio
import numpy as np
import pickle
 
conf = config.get_config()

def __load_data( dir ):
    x = []
    y = []
    print( dir )
    num = len( os.listdir( dir ) )
    for i, file in enumerate( os.listdir( dir ) ):
        if not file.endswith( '.jpg' ):
            continue
        x.append( imageio.imread( os.path.join( dir, file ) ) )
        y.append( 1 if file.startswith( 'car' ) else 0 )
        sys.stdout.write('\r{}/{}'.format( i,num ) )
    
    return np.asarray( x ), np.asarray( y )

def load_data():
    cowc_dir = conf[ 'datasets' ][ 'cowc_dir' ]
    
    if os.path.exists( os.path.join( cowc_dir, 'cowc.bin' ) ):
            with open( os.path.join( cowc_dir, 'cowc.bin' ), 'rb' ) as f:
                return pickle.load( f )
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for loc_dir in os.listdir( cowc_dir ):
        loc_dir = os.path.join( cowc_dir, loc_dir )
        if not os.path.isdir( loc_dir ):
            continue
        x_, y_ = __load_data( os.path.join( loc_dir, 'test' ) )
        x_test.append( x_ )
        y_test.append( y_ ) 
        
        x_, y_ = __load_data( os.path.join( loc_dir, 'train' ) )
        x_train.append( x_ )
        y_train.append( y_ )
        
    x_train = np.vstack( x_train )
    y_train = np.concatenate( y_train )
    
    x_test = np.vstack( x_test )
    y_test = np.concatenate( y_test )
    
    with open( os.path.join( cowc_dir, 'cowc.bin' ), 'wb' ) as f:
        pickle.dump( ((x_train, y_train), (x_test, y_test)) ,f )
    
    return (x_train, y_train), (x_test, y_test) 


def write_binary( data, type, file ):
    with open( file , 'wb' ) as f:
        f.write( data.astype( type ).tobytes() )

def export_binary():
    (x_train, y_train), (x_test, y_test) = load_data()
    cowc_dir = conf[ 'datasets' ][ 'cowc_dir' ]
    write_binary( x_train, np.uint8 ,os.path.join( cowc_dir, 'cowc_train_x.bin' ) )
    write_binary( x_test, np.uint8 ,os.path.join( cowc_dir, 'cowc_test_x.bin' ) )
    
    write_binary( y_train, np.uint8 ,os.path.join( cowc_dir, 'cowc_train_y.bin' ) )
    write_binary( y_test, np.uint8 ,os.path.join( cowc_dir, 'cowc_test_y.bin' ) )
    
    
    
    
    
    
    
    
    
    
