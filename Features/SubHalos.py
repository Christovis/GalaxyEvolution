import sys
import h5py
import numpy as np
import pandas as pd
#from ReadTree import DHaloReader as DHalo
import ReadTree
sys.path.insert(0,'../../lib/')
import read_hdf5
import read_hdf5_eagle


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
    def __init__(self, simulation, snapnum, nepoch):
        """ 
        """
        self.simulation = simulation

        if (self.simulation == "christian_dm"):
            self.sh_orig_file = "/cosma5/data/dp004/dc-oles1/dhalo/data/GR/"
            self.sh_reor_file = "/cosma5/data/dp004/dc-beck3/Galaxy_Evolution/" \
                                "SubFind/dm_only/L62_N512_GR/subfind.0.hdf5"
            self.mt_file = "/cosma5/data/dp004/dc-oles1/dhalo/out/trees/GR/" \
                           "treedir_075/tree_075.0.hdf5"
            self.snapshot = read_hdf5.snapshot(snapnum, self.sh_orig_file)
            print('Loading snapshot at z=%f' % self.snapshot.header.redshift)
        elif (self.simulation == 'christian_fp'):
            print('Not Ready !!!')
        elif (self.simulation == "EAGLE"):
            self.sh_orig_file = "/cosma5/data/Eagle/ScienceRuns/Planck1/" \
                                "L0100N1504/PE/REFERENCE/data/"
            self.sh_reor_file = "."
            self.mt_file = "/gpfs/data/Eagle/yanTestRuns/MergerTree/Dec14/" \
                           "L0100N1504/EAGLE_L0100N1504_db.hdf5"
            ReadTree.read_tree_keys(self.mt_file)
            snapbases = '/eagle_subfind_particles_0%s_z000p000'%str(snapnum)
            dirbases = 'particledata_0%s_z000p000'%str(snapnum)
            self.snapshot = read_hdf5_eagle.snapshot(snapnum, self.sh_orig_file,
                                                     snapbases=[snapbases],
                                                     dirbases=[dirbases])
        
        # Load the useful fields
        if (self.simulation == 'EAGLE'):
            print('self.snapshot', self.snapshot)
            self.snapshot.group_catalog(['GroupMass', 'Group_M_Crit200',
                                         'Group_R_Crit200', 'NumOfSubhalos',
                                         'GroupLength', 'GroupCentreOfPotential',
                                         'MassType'])
            ## Read with respect to the merger-tree re-ordered SubFind file
            hdf = h5py.File(self.sh_reor_file, 'r')
            self.df = pd.DataFrame({'snapnum' : hdf['Subhalo']['SnapNum'][:],
                                    'mass_total' : hdf['Subhalo']['SubhaloMass'][:],
                                    'halfmassrad' : hdf['Subhalo']['SubhaloHalfmassRad'][:],
                                    'sigma' : hdf['Subhalo']['SubhaloVelDisp'][:],
                                    'nodeIndex' : hdf['Subhalo']['nodeIndex'][:],
                                    'id_mostbound' : hdf['Subhalo']['SubhaloIDMostbound'][:]})
            _indx = self.df[self.df['snapnum'] == snapnum].index
            self.df = self.df[self.df['snapnum'] == snapnum]
            self.df.index = range(len(self.df.index))
        else:
            # First Hand Data
            ## Snapshot particle related data
            self.snapshot.group_catalog(["SubhaloLenType",
                                         "SubhaloPos",
                                         "SubhaloIDMostbound"])
            dfpart = pd.DataFrame(
                    {'id_mostbound' : (self.snapshot.cat['SubhaloIDMostbound']).astype(np.int64),
                     'X' : (self.snapshot.cat['SubhaloPos'][:, 0]).astype(np.float64),
                     'Y' : (self.snapshot.cat['SubhaloPos'][:, 1]).astype(np.float64),
                     'Z' : (self.snapshot.cat['SubhaloPos'][:, 2]).astype(np.float64),
                     'n_part' : (self.snapshot.cat["SubhaloLenType"][:, 1]).astype(np.int64)}
                    )
            subhalo_offset = (np.cumsum(dfpart['n_part'].values) - \
                              dfpart['n_part'].values).astype(int)
            dfpart['subhalo_offset'] = pd.Series(subhalo_offset, index=dfpart.index, dtype=int)
            self.snapshot.read(["Coordinates"], parttype=[1])
            del subhalo_offset, self.snapshot.cat

            ## Read with respect to the merger-tree re-ordered SubFind file
            hdf = h5py.File(self.sh_reor_file, 'r')
            self.df = pd.DataFrame({'snapnum' : hdf['Subhalo']['SnapNum'][:],
                                    'mass_total' : hdf['Subhalo']['SubhaloMass'][:],
                                    'halfmassrad' : hdf['Subhalo']['SubhaloHalfmassRad'][:],
                                    'sigma' : hdf['Subhalo']['SubhaloVelDisp'][:],
                                    'nodeIndex' : hdf['Subhalo']['nodeIndex'][:],
                                    'id_mostbound' : hdf['Subhalo']['SubhaloIDMostbound'][:]})
            _indx = self.df[self.df['snapnum'] == snapnum].index
            self.df = self.df[self.df['snapnum'] == snapnum]
            self.df.index = range(len(self.df.index))

            spin = hdf['Subhalo']['SubhaloSpin'][:]
            spin = spin[_indx, :]
            spin = [np.sqrt((spin[ii, 0]**2 + spin[ii, 1]**2 + spin[ii, 2]**2)/3) for ii in range(len(spin))]
            self.df['spin'] = pd.Series(spin, index=self.df.index, dtype=float)
            
            ## Filter
            _indx = self.match_halos(self.df['id_mostbound'].values, dfpart['id_mostbound'].values)
            self.df = self.df.iloc[_indx]
            self.df.index = range(len(self.df.index))

            _indx = self.match_halos(dfpart['id_mostbound'].values, self.df['id_mostbound'].values)
            dfpart = dfpart.iloc[_indx]
            dfpart.index = range(len(dfpart.index))

            # Second Hand Data
            self.principal_axis(dfpart)
            self.progenitors(snapnum-nepoch, snapnum)


    def match_halos(self, a, b):
        idxa = np.argsort(a)
        sorteda = a[idxa]
        idxb = np.searchsorted(sorteda, b)
        return idxb


    def principal_axis(self, dfpart):
        """ Compute triaxial halo shapes through principal axis of ellipsoids
        """
        _ellipticity = np.zeros(self.df['nodeIndex'].size)
        _prolateness = np.zeros(self.df['nodeIndex'].size)
        for i in range(self.df['nodeIndex'].size):
            if dfpart['n_part'][i] < 100:
                continue
            _coord = self.snapshot.data['Coordinates']['dm'][
                    dfpart['subhalo_offset'][i] : \
                            (dfpart['subhalo_offset'][i] + \
                             dfpart['n_part'][i]), :]
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
            _tau = _a + _b + _c
            _ellipticity[i] = (_a - _b) / (2*_tau)
            _prolateness[i] = (_a - 2*_b + _c) / (2*_tau)

        #self.sphericity = self.c/self.a
        #self.elongation = self.b/self.a
        #self.triaxality = (self.a **2 - self.b**2) / (self.a**2 - self.c**2)
        self.df['ellipticity'] = pd.Series(_ellipticity, index=self.df.index, dtype=float)
        self.df['prolateness'] = pd.Series(_prolateness, index=self.df.index, dtype=float)
        del _ellipticity, _prolateness
        del dfpart
    
    
    def progenitors(self, snapnum_pred, snapnum_obs):
        """
        """
        #if self.simulation == "christian_dm":
        mtree = DHalo(self.mt_file)
        _nodeID, _prognum = mtree.find_progenitors_until_z(
                mtree, self.df['nodeIndex'].values, snapnum_pred, snapnum_obs)
        ## Filter
        _indx = self.match_halos(self.df['nodeIndex'].values, _nodeID)
        
        
        print('test 1', len(_indx), len(np.unique(_indx)))
        print('test 2', self.df['nodeIndex'].size,
              np.max(_indx), self.df.index)
        print(np.sort(_nodeID))
        print(np.sort(self.df['nodeIndex'].values))
        
        self.df = self.df.iloc[np.unique(_indx)]
        self.df.index = range(len(self.df.index))
        
        print(np.sort(self.df['nodeIndex'].values))

        _indx = self.match_halos(_nodeID, self.df['nodeIndex'].values)
        _nodeID = _nodeID[_indx]
        _prognum = _prognum[_indx]

        self.df = self.df.sort_values(by=['nodeIndex'], ascending=True)
        _indx = np.argsort(_nodeID)

        _prognum_dict = {}
        for ii in range(_prognum.shape[1]):
            _prognum_dict[('progenitors', str(ii))] = _prognum[_indx, ii]
        dfp = pd.DataFrame.from_dict(_prognum_dict)
        self.df = pd.concat([self.df, dfp], axis=1)               
        #if self.simulation == "EAGLE":


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





		

