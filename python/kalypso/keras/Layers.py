import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class SplitLayer( tf.keras.layers.Layer ):
  """
  Split the input of the layer along the given axis into equall chunks
  """

  def __init__( self, n_splits=1, split_axis=1, overlap=0, **kwargs ):
    """
    Split the input of the layer along the given axis into equall chunks
    
    `n_splits` needs to divide the input the shape along `split_axis` without 
    remainder.

    `overlap` adds `overlap` elements to the start and end of every split. The 
    number of elements along the splitaxis will be 
    input_shape[ split_axis ] / n_splits + 2 * overlap.
    For the first/last element 2 * `overlap` items will be added and the 
    end/beginning since there is no prior/following split.

    """
    self.n_splits = n_splits
    self.split_axis = split_axis
    self.overlap = overlap
    super( SplitLayer, self ).__init__( **kwargs )

  def call( self, x ):
    if self.overlap == 0:
      sub_tensors = tf.split( x, self.n_splits, axis=self.split_axis )
    else:
      split_size = int( x.shape[ self.split_axis ] / self.n_splits + 2 * self.overlap )
      # generate the begin and size
      begin = [ 0 ] * len( x.shape )  # begin is the offset from wich we start
      size = [ -1 ] * len( x.shape )  # size is the number of elements we take from each axis.
                                      # -1 means all.
      size[ self.split_axis ] = split_size
      # start with the first split
      sub_tensors = [ tf.slice( x, begin=begin, size=size ) ]
      # do all the splits except for the last
      for i in range( 1, self.n_splits - 1 ):
        begin[ self.split_axis ] += int( x.shape[ self.split_axis ] / self.n_splits - self.overlap )
        sub_tensors.append( tf.slice( x, begin=begin, size=size ) )
      # do the last split
      begin[ self.split_axis ] = int( x.shape[ self.split_axis ] - split_size )
      sub_tensors.append( tf.slice( x, begin=begin, size=size ) )
    return sub_tensors

  def compute_output_shape( self, input_shape ):
    sub_tensor_shape = list( input_shape )
    sub_tensor_shape[ self.split_axis ] = input_shape[ self.split_axis ] / self.n_splits + 2 * self.overlap
    sub_tensor_shape = tuple( sub_tensor_shape )
    list_of_output_shape = [ sub_tensor_shape ] * self.n_splits
    return  list_of_output_shape

  def compute_mask( self, inputs, mask=None ):
    return self.n_splits * [ None ]

  def get_config( self ):
    conf = super( SplitLayer, self ).get_config()
    conf[ 'split_axis' ] = self.split_axis
    conf[ 'n_splits' ] = self.n_splits
    conf[ 'overlap' ] = self.overlap
    return conf


class PolyActivation( object ):
  
    def __init__( self, a=1, b=0, c=0, d=0, **kwargs ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def get_config( self ):
        conf = {}
        conf[ 'a' ] = self.a
        conf[ 'b' ] = self.b
        conf[ 'c' ] = self.c
        conf[ 'd' ] = self.d
        return conf

    @classmethod
    def from_config( cls, config ):
        return cls( **config )
    
    def __call__( self, x ):
        return self.a * K.pow( x, 3 ) + self.b * K.pow( x, 2 ) + self.c * x + self.d
