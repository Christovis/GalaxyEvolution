import sys
import h5py
import numpy as np
from ReadTree import DHaloReader as DHalo
sys.path.insert(0,'../../lib/')
import read_hdf5


def read_subhalo_keys(fname, snapshot):
    """ Print keys of re-ordered SubFind subhalos output """
    df = h5py.File(fname, 'r')
    print('The following Subhalo properties are available:')
    print('\n'.join(list(df['Subhalo'].keys())))
    print('\n')


def read_group_keys(fname, snapshot):
    df = h5py.File(fname, 'r')
    print('The following Group properties are available:')
    print('\n'.join(list(df.keys())))
    print('\n')


class SubHalos:
    def __init__(self, simulation, snapshot):
        """ 
        """
        self.sh_orig_file = sh_orig_file
        self.sh_reor_file = sh_reor_file
        self.mt_file = mt_file

        self.simulation = simulation

		if (self.simulation == 'christian_dm'):
			self.sh_orig_file = '/cosma6/data/dp004/dc-arno1/SZ_project/dark_matter_only/L62_N512_GR/'
            self.sh_reor_file = '/cosma5/data/dp004/dc-beck3/Galaxy_Evolution/SubFind/dm_only/L62_N512_GR/subfind.0.hdf5'
            self.mt_file = '/cosma5/data/dp004/dc-oles1/dhalo/out/trees/GR/treedir_075/tree_075.0.hdf5'
			self.snapshot = read_hdf5.snapshot(snapshot, h5_dir, snapbases = ['/gadget-groupordered_'])
        elif (self.simulation == 'christian_fp'):
            print('Not Ready !!!')
			#self.sh_orig_file = '/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_GR_kpc/'
            #self.sh_orig_file = '' #not available yet
			#self.snapshot = read_hdf5.snapshot(snapshot, h5_dir, snapbases = ['/gadget-groupordered_'])
		elif (self.simulation == 'EAGLE'):
			h5_dir = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'
			snapbases = '/eagle_subfind_particles_0%s_z000p000'%str(snapshot)
			dirbases = 'particledata_0%s_z000p000'%str(snapshot)
			self.snapshot = read_hdf5_eagle.snapshot(snapshot, h5_dir,
                                                     snapbases=[snapbases],
                                                     dirbases=[dirbases])

		# Load the useful fields
		if(self.simulation == 'EAGLE'):
			self.snapshot.group_catalog(['GroupMass', 'Group_M_Crit200',
                                         'Group_R_Crit200', 'NumOfSubhalos',
                                         'GroupLength', 'GroupCentreOfPotential',
                                         'MassType'])
		else:
			self.snapshot.group_catalog(['GroupMass', 'Group_M_Crit200',
                                         'Group_R_Crit200', 'GroupMassType',
                                         'GroupNsubs', 'GroupLenType', 'GroupPos',
                                         'GroupCM', 'SubhaloMassType',
                                         'SubhaloMass', 'SubhaloVelDisp',
                                         'SubhaloVmax','GroupFirstSub',
                                         'SubhaloHalfmassRadType','GroupPos',
                                         'SubhaloVmaxRad'])


    def get_orig_subfind_data(self, snapshot):
        # Read SubFind file
        self.snapshot = read_hdf5.snapshot(snapshot, self.sh_orig_file)
        self.snapshot.group_catalog(["SubhaloLenType",
                                     "SubhaloPos",
                                     "SubhaloIDMostbound"])
        self.n_part = (self.snapshot.cat["SubhaloLenType"][:, 1]).astype(np.int64)
        self.subhalo_offset = (np.cumsum(self.n_part) - self.n_part).astype(int)
       
        self.snapshot.read(["Coordinates"], parttype=[1])
        #self.snapshot.read(["Coordinates", "Velocities"], parttype=[1])
        #self.particle_coordinates = self.snapshot.data['Coordinates']['dm']  #*scale
        #del self.snapshot.data['Coordinates']['dm']
        del self.snapshot.cat["SubhaloLenType"]

    
    def get_reor_subfind_data(self, snapshot):
        # Read with respect to the merger-tree re-ordered SubFind file
        df = h5py.File(self.sh_reor_file, 'r')
        indx = np.where(df['Subhalo']['SnapNum'][:] == snapshot)
        
        mass_total = df['Subhalo']['SubhaloMass'][:]
        self.mass_total = mass_total[indx]
       
        halfmassrad = df['Subhalo']['SubhaloHalfmassRad'][:]
        self.halfmassrad = halfmassrad[indx]
        
        sigma = df['Subhalo']['SubhaloVelDisp'][:]
        self.sigma = sigma[indx]
        
        nodeIndex = df['Subhalo']['nodeIndex'][:]
        self.nodeIndex = nodeIndex[indx]
        
        spin = df['Subhalo']['SubhaloSpin'][:, :]
        spin = spin[indx, :][0]
        spin = [np.sqrt((spin[ii, 0]**2 + spin[ii, 1]**2 + spin[ii, 2]**2)/3) for ii in range(len(spin))]
        spin = np.asarray(spin)
        self.spin = spin
        
        id_mostbound = df['Subhalo']['SubhaloIDMostbound'][:]
        self.filter_subhalos(id_mostbound[indx], self.subhalo_id)
        
        #TODO: Include mass of FoF-Group/Cluster
        #self.concentration = concentration()
        #self.environment = environment()


    def filter_subhalos(self, pre_filter_ids, post_filter_ids):
        """ Filter Subhalos in feature list, in order to keep it complete
        """
        # Why does this not work? Much handier
        #_intersect1d, _indx_pre, _indx_post = np.intersect1d(pre_filter_ids,
        #                                                     post_filter_ids,
        #                                                     assume_unique=False,
        #                                                     return_indices=True)
        _indx_pre = np.arange(pre_filter_ids.shape[0])[np.in1d(pre_filter_ids,
                                                               post_filter_ids,
                                                               assume_unique=False)]
        _indx_post = np.arange(post_filter_ids.shape[0])[np.in1d(post_filter_ids,
                                                                 pre_filter_ids,
                                                                 assume_unique=False)]
        # Match data entries
        _pre_sort_indx = np.argsort(pre_filter_ids[_indx_pre])
        _post_sort_indx = np.argsort(post_filter_ids[_indx_post])
    
        for k in self.__dict__.keys():
            if k in ['sh_orig_file', 'sh_reor_file', 'mt_file', 'snapshot']:
                continue
            elif not k.startswith('__'):
                if k in ['ellipticity', 'prolateness', 'subhalo_id']:
                    v = getattr(self, k)
                    setattr(self, k, v[_indx_post[_post_sort_indx]])
                else:
                    v = getattr(self, k)
                    setattr(self, k, v[_indx_pre[_pre_sort_indx]])


    def principal_axis(self):
        """ Compute triaxial halo shapes through principal axis of ellipsoids
        """
        _subhalo_id = []
        _ellipticity = []
        _prolateness = []
        for i in range(len(self.n_part)):
            if self.n_part[i] < 50:
                continue
            _coord = self.snapshot.data['Coordinates']['dm'][
                    self.subhalo_offset[i] : self.subhalo_offset[i]+self.n_part[i], :]
            _centre = [_coord[:, 0].min() + (_coord[:, 0].max() - _coord[:, 0].min())/2,
                       _coord[:, 1].min() + (_coord[:, 1].max() - _coord[:, 1].min())/2,
                       _coord[:, 2].min() + (_coord[:, 2].max() - _coord[:, 2].min())/2]

            # Distance to parent halo
            _distance =  _coord - _centre 
            
            # Distance weighted Intertia Tensor / Reduced Inertia Tensor
            _I = np.dot(_distance.transpose(), _distance)
            _I /= np.sum(_distance**2)
            _I[~(np.eye(3) == 1)] *= -1
    
            _eigenvalues, _eigenvectors = np.linalg.eig(_I)
            if ((_eigenvalues < 0).sum() > 0) or (np.sum(_eigenvalues) == 0):
                continue
            _eigenvalues = np.sqrt(_eigenvalues)
            _c, _b, _a = np.sort(_eigenvalues)
            _subhalo_id.append(int(self.snapshot.cat['SubhaloIDMostbound'][i]))
            _tau = _a + _b + _c
            _ellipticity.append((_a - _b) / (2*_tau))
            _prolateness.append((_a - 2*_b + _c) / (2*_tau))

        #self.sphericity = self.c/self.a
        #self.elongation = self.b/self.a
        #self.triaxality = (self.a **2 - self.b**2) / (self.a**2 - self.c**2)
        _subhalo_id = np.asarray(_subhalo_id)
        self.ellipticity = np.asarray(_ellipticity)
        self.prolateness = np.asarray(_prolateness)
        self.subhalo_id = np.asarray(_subhalo_id)
        del _ellipticity, _prolateness, _subhalo_id
        del self.snapshot.data['Coordinates']['dm']
        del self.snapshot.cat['SubhaloPos']
        del self.snapshot.cat['SubhaloIDMostbound']
    
    
    def progenitors(self, snapnum_obs, snapnum_pred):
        """
        """
        mtree = DHalo(self.mt_file)
        nodeID, prognum = mtree.find_progenitors_until_z(
                self, mtree, snapnum_pred, snapnum_obs)
        _indx_pre = np.arange(self.nodeIndex.shape[0])[np.in1d(self.nodeIndex,
                                                               nodeID,
                                                               assume_unique=False)]
        _indx_post = np.arange(nodeID.shape[0])[np.in1d(nodeID,
                                                        self.nodeIndex,
                                                        assume_unique=False)]
        post_sort_indx = np.argsort(nodeID[_indx_post])
        pre_sort_indx = np.argsort(self.nodeIndex[_indx_pre])
        for k in self.__dict__.keys():
            if k in ['sh_orig_file', 'sh_reor_file', 'mt_file', 'snapshot']:
                continue
            elif not k.startswith('__'):
                v = getattr(self, k)
                setattr(self, k, v[_indx_pre[pre_sort_indx]])
        self.prognum = prognum[_indx_post[post_sort_indx]]


    def concentration(self):
        self.concentration = np.zeros(self.N_halos)
        _units = self.snapshot.const.Mpc/1000./np.sqrt(0.001)
        v200c = _units*np.sqrt(self.snapshot.const.G*self.m200c/self.r200c)

        def f_to_solve(c, vmax, v200c):
            return vmax/v200c - np.sqrt(0.216*c/(np.log(1+c) - c/(1+c)))

        for i in range(self.N_halos):
            self.concentration[i] = fsolve(f_to_solve, 0.5,args=(self.vmax[i], v200c[i]))


    def environment(self, f):
        """
        Quantify the environment of subhalos based on: https://arxiv.org/pdf/1103.0547.pdf
        """
        self.haas_env = np.zeros(self.N_halos)
        def closest_node(node, nodes):
            return distance.cdist([node],nodes).argmin()

        for i in range(self.N_halos):
            halopos_neighbors = self.halopos[self.m200c > f * self.m200c[i]]
            if(halopos_neighbors.shape[0] == 0):
                self.haas_env[i] = -1
                continue

            index = closest_node(self.halopos[i], halopos_neighbors)
            
            distance_fneigh = np.linalg.norm( self.halopos[i] - halopos_neighbors[index])
            self.haas_env[i] = distance_fneigh / self.r200c[ self.m200c > f *self.m200c[i]][index] 





		

