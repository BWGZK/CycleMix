import math

def cyclic_learning_rate(global_step,learning_rate=1e-5,max_lr=1e-4,step_size=20.,gamma=0.99994,mode='triangular',name=None):
  cycle = math.floor( 1 + global_step / ( 2 * step_size ) )
  x = abs( global_step / step_size - 2 * cycle + 1 )
  clr = learning_rate + ( max_lr - learning_rate ) * max( 0, 1 - x )
  return clr