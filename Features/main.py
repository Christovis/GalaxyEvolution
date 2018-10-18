from __future__ import division
import numpy as np
import h5py
#from ReadTree import DHaloReader as DHalo
#from ReadTree import read_tree_keys
from SubHalos import SubHalos as SHfuncs
#from SubHalos import read_subhalo_keys
#from SubHalos import read_group_keys

#******************** Load Data ***********************
# snapnum:redshift -> 71:0.131 -> 64:0.393

## SubFind

SH = SHfuncs('EAGLE', 28, 3)
SH.df.to_hdf('EAGLE_contin.h5', key='Halos', mode='w')

## Merger Tree
#print('prognum', SH.prognum[:10, :])
##print('Nr of Subhalo', len(SH.subhalo_id), len(SH.prognum))
#
#hf = h5py.File('SubhaloData_contin.h5', 'w')
#hf.create_dataset('M200', data=SH.mass_total)
#hf.create_dataset('VelDisp', data=SH.sigma)
#hf.create_dataset('Spin', data=SH.spin)
#hf.create_dataset('HalfmassRad', data=SH.halfmassrad)
#hf.create_dataset('Ellipticity', data=SH.ellipticity)
#hf.create_dataset('Prolateness', data=SH.prolateness)
#hf.create_dataset('Progenitor_number', data=SH.prognum)
#hf.create_dataset('SubFindID', data=SH.subhalo_id)
#hf.create_dataset('MTreeID', data=SH.nodeIndex)
#hf.close()

#hf = h5py.File('SubhaloData_training.h5', 'w')
#hf = h5py.File('SubhaloData_validation.h5', 'w')


