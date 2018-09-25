from __future__ import division
import numpy as np
import h5py
from ReadTree import DHaloReader as DHalo
from ReadTree import read_tree_keys
from SubHalos import SubHalos as SHfuncs
from SubHalos import read_subhalo_keys
from SubHalos import read_group_keys

#******************** Load Data ***********************
# snapnum:redshift -> 71:0.131 -> 64:0.393

## SubFind
sh_orig_file = '/cosma6/data/dp004/dc-arno1/SZ_project/dark_matter_only/L62_N512_GR/'
sh_reor_file = '/cosma5/data/dp004/dc-beck3/Galaxy_Evolution/SubFind/dm_only/L62_N512_GR/subfind.0.hdf5'
mt_file = '/cosma5/data/dp004/dc-oles1/dhalo/out/trees/GR/treedir_075/tree_075.0.hdf5'  # 71 is 41, simply because

SH = SHfuncs(sh_orig_file, sh_reor_file, mt_file)

SH.get_orig_subfind_data(41) 
SH.principal_axis()
SH.get_reor_subfind_data(71)
print('Nr of Subhalo', len(SH.subhalo_id))

## Merger Tree
SH.progenitors(71, 65)
print('prognum', SH.prognum[:10, :])
#print('Nr of Subhalo', len(SH.subhalo_id), len(SH.prognum))

hf = h5py.File('SubhaloData_contin.h5', 'w')
hf.create_dataset('M200', data=SH.mass_total)
hf.create_dataset('VelDisp', data=SH.sigma)
hf.create_dataset('Spin', data=SH.spin)
hf.create_dataset('HalfmassRad', data=SH.halfmassrad)
hf.create_dataset('Ellipticity', data=SH.ellipticity)
hf.create_dataset('Prolateness', data=SH.prolateness)
hf.create_dataset('Progenitor_number', data=SH.prognum)
hf.create_dataset('SubFindID', data=SH.subhalo_id)
hf.create_dataset('MTreeID', data=SH.nodeIndex)
hf.close()

#hf = h5py.File('SubhaloData_training.h5', 'w')
#hf = h5py.File('SubhaloData_validation.h5', 'w')


