import numpy as np
from . import cosmology

class Galaxy(object):
    """
    Def. This class represents a galaxy, containing different components such as stellar and gas particles.

    Attributes
    ----------
    name : str (optional, default='gal')
        Name of the galaxy to be used for storage.
    stars : dict #TODO
    gas : dict #TODO
    position : np.array (optional)
        (x, y, z) np.array vector corresponding to the position of the galaxy in the same frame as the particles.
        If not provided, the particles position vector will be assumed to be in the galaxy frame.
    velocity :
        (v_x, v_y, v_z) np.array vector corresponding to the velocity of the galaxy in the same frame as the particles.
        If not provided, all particles will be assumed to be in the galaxy rest frame.
    spin : np.array
        (L_x, L_y, L_z) np.array vector corresponding to the spin of the galaxy in the same frame as the particles.
        If not provided, all particles will be assumed to be (0, 0, 0).
    kernel : smooth_kernel.KernelObject (optional, default=GaussianKernel(mean=0, sigma=.3)
        Kernel used to smooth the distribution of particles.

    Methods
    -------
    build_stars
    set_galaxy_to_rest_frame
    project_galaxy
    """
    stars = {}
    gas = {}

    def __init__(self, **kwargs):

        # Particles
        stellar_particles = kwargs.get('stars', None)
        self.build_stars(stellar_particles)
        gas_particles = kwargs.get('gas', None)
        self.build_gas(gas_particles)

        # Galaxy params
        self.name = kwargs.get('name', "gal")
        self.spin = kwargs.get('gal_spin',
                               np.full(3, fill_value=np.nan))  # kpc/(km/s)
        self.velocity = kwargs.get('gal_vel',
                                   np.full(3, fill_value=np.nan))  # km/s
        self.position = kwargs.get('gal_pos',
                                   np.full(3, fill_value=np.nan))  # kpc


    def build_stars(self, particles):
        """todo."""
        if particles is not None:
            for key in list(particles.keys()):
                self.stars[key] = particles[key][()]
            # Compute stellar particle age
            ages = np.full(self.stars['GFM_StellarFormationTime'].size,
                           fill_value=np.nan)
            wind = self.stars['GFM_StellarFormationTime'] < 0
            ages[~wind] = np.interp(
                self.stars['GFM_StellarFormationTime'][~wind],
                cosmology.scale_f[::-1], cosmology.age_f[::-1])
            self.stars['ages'] = ages
            self.stars['wind'] = wind
        else:
            self.stars = None

    def build_gas(self, particles):
        """todo."""
        if particles is not None:
            for key in list(particles.keys()):
                self.gas[key] = particles[key][()]
            # Gas temperature
            if 'ElectronAbundance' in particles.keys():
                self.gas['temp'] = compute_gas_temp(self.gas)
        else:
            self.gas = None

    def set_to_galaxy_rest_frame(self):
        """Set the position and velocity of gas and stellar particles to the galaxy rest frame."""
        print("[Galaxy] Setting particles to galaxy rest-frame")
        if not np.isfinite(self.position).all():
            print(" [Galaxy] Incorrect galaxy position: ({} {} {})"
                  .format(*self.position))
            return
        if not np.isfinite(self.velocity).all():
            print(" [Galaxy] Incorrect galaxy velocity: ({} {} {})"
                  .format(*self.velocity))
            return

        if self.stars is not None:
            self.stars['Velocities'] += -self.velocity[np.newaxis, :]
            self.stars['Coordinates'] += -self.position[np.newaxis, :]
            print(" [Galaxy] Stellar particles in rest-frame")
        if self.gas is not None:
            self.gas['Velocities'] += -self.velocity[np.newaxis, :]
            self.gas['Coordinates'] += -self.position[np.newaxis, :]
            print(" [Galaxy] Gas particles in rest-frame")
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)

    def proyect_galaxy(self, orthogonal_vector=None):
        """
        Compute the projected the positions and velocities of all galaxy elements.

        If no vector is provided, the particles will be projected to the XY plane.

        Parameters
        ----------
        orthogonal_vector: np.array (default=(0, 0, 1))
            Orthogonal vector to the projected plane.
        """
        if orthogonal_vector is None:
            if self.stars is not None:
                self.stars['ProjCoordinates'] = self.stars['Coordinates'].copy().T
                self.stars['ProjVelocities'] = self.stars['Velocities'].copy().T
            if self.gas is not None:
                self.gas['ProjCoordinates'] = self.gas['Coordinates'].copy().T
                self.gas['ProjVelocities'] = self.gas['Velocities'].copy().T
            self.proyection_vector = (None, None, None)
        else:
            norm = orthogonal_vector / np.sqrt(sum(orthogonal_vector**2))
            self.proyection_vector = norm
            b = np.cross(np.array([0, 0, 1]), norm)
            b /= np.sqrt(sum(b**2))
            theta = np.arccos(np.dot(np.array([0, 0, 1]), norm))
            q0, q1, q2, q3 = (np.cos(theta/2), np.sin(theta/2)*b[0],
                              np.sin(theta/2)*b[1], np.sin(theta/2)*b[2])
            # Quartenion matrix
            Q = np.zeros((3, 3))
            Q[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
            Q[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
            Q[2, 2] = q0**2 - q1**2 - q2**2 + q3**2
            Q[0, 1] = 2*(q1*q2 - q0*q3)
            Q[0, 2] = 2*(q1*q3 + q0*q2)
            Q[1, 0] = 2*(q1*q2 + q0*q3)
            Q[1, 2] = 2*(q2*q3 - q0*q1)
            Q[2, 0] = 2*(q1*q3 - q0*q2)
            Q[2, 1] = 2*(q3*q2 + q0*q1)
            # New basis
            u, v, w = (np.dot(Q, np.array([1, 0, 0])),
                       np.dot(Q, np.array([0, 1, 0])),
                       np.dot(Q, np.array([0, 0, 1])))
            # Change of basis matrix
            W = np.array([u, v, w])
            inv_W = np.linalg.inv(W)
            if self.stars is not None:
                self.stars['ProjCoordinates'] = np.array([
                    np.sum(self.stars['Coordinates'] * u[np.newaxis, :], axis=1),
                    np.sum(self.stars['Coordinates'] * v[np.newaxis, :], axis=1),
                    np.sum(self.stars['Coordinates'] * w[np.newaxis, :], axis=1)
                    ])
                self.stars['ProjVelocities'] = np.array([
                    np.sum(self.stars['Velocities'] * inv_W[0, :][np.newaxis, :],
                           axis=1),
                    np.sum(self.stars['Velocities'] * inv_W[1, :][np.newaxis, :],
                           axis=1),
                    np.sum(self.stars['Velocities'] * inv_W[2, :][np.newaxis, :],
                           axis=1)
                    ])
            if self.gas is not None:
                self.gas['ProjCoordinates'] = np.array([
                    np.sum(self.gas['Coordinates'] * u[np.newaxis, :], axis=1),
                    np.sum(self.gas['Coordinates'] * v[np.newaxis, :], axis=1),
                    np.sum(self.gas['Coordinates'] * w[np.newaxis, :], axis=1)
                    ])
                self.gas['ProjVelocities'] = np.array([
                    np.sum(self.gas['Velocities'] * inv_W[0, :][np.newaxis, :],
                           axis=1),
                    np.sum(self.gas['Velocities'] * inv_W[1, :][np.newaxis, :],
                           axis=1),
                    np.sum(self.gas['Velocities'] * inv_W[2, :][np.newaxis, :],
                           axis=1)
                    ])

    def get_galaxy_pos(self, stars=True, gas=False):
        pos = np.zeros((0, 3))
        mass = np.zeros(0)
        if stars:
            if self.stars is not None:
                pos = np.vstack((pos, self.stars['Coordinates']))
                mass = np.hstack((mass, self.stars['Masses']))
        if gas:
            if self.gas is not None:
                pos = np.vstack((pos, self.gas['Coordinates']))
                mass = np.hstack((mass, self.gas['Masses']))
        self.position = (
            np.sum(pos * mass[:, np.newaxis], axis=0) / np.sum(mass))
        print("[Galaxy] New galaxy position: ", self.position)
    
    def get_galaxy_vel(self, stars=True, gas=False):
        vel = np.zeros((0, 3))
        mass = np.zeros(0)
        if stars:
            if self.stars is not None:
                vel = np.vstack((vel, self.stars['Velocities']))
                mass = np.hstack((mass, self.stars['Masses']))
        if gas:
            if self.gas is not None:
                vel = np.vstack((vel, self.gas['Velocities']))
                mass = np.hstack((mass, self.gas['Masses']))
        self.velocity = (
            np.sum(vel * mass[:, np.newaxis], axis=0) / np.sum(mass))
        print("[Galaxy] New galaxy velocity: ", self.velocity)

    def get_stellar_map(self, out, stat_val, statistic='sum'):
        """..."""
        if statistic == 'sum':
            for xbin, ybin, val in zip(self.stars['xbin'], self.stars['ybin'],
                                       stat_val):
                if (xbin < 0) | (ybin < 0):
                    continue
                out[xbin, ybin] += val
        elif statistic == 'mean':
            count = np.zeros(out.shape, dtype=int)
            for xbin, ybin, val in zip(self.stars['xbin'], self.stars['ybin'],
                                       stat_val):
                if (xbin < 0) | (ybin < 0):
                    continue
                out[xbin, ybin] += val
                count[xbin, ybin] += 1
            out = out / (count + 1e-10)
        elif statistic == 'std':
            occupated_bins = np.unique(
                np.array([self.stars['xbin'], self.stars['ybin']]).T,
                axis=0)
            for oc_bin in occupated_bins:
                mask = (self.stars['xbin'] == oc_bin[0]
                        ) & (self.stars['ybin'] == oc_bin[1])
                out[oc_bin[0], oc_bin[1]] = np.std(stat_val[mask])
        return out

class TNG_Galaxy(Galaxy):
    def __init__(self, h5py_file):
        stars = h5py_file["PartType4"]
        gas =  h5py_file["PartType0"]
        super().__init__(stars=stars, gas=gas)

class EAGLE_Galaxy(Galaxy):
    def __init__(self, **kwargs):
        raise NotImplementedError("Class not implemented")
        ### Do stuff
        #super().__init__(**kwargs)


def compute_gas_temp(gas_file, gamma=1.6667):
    """Compute the temperature of gas particles."""
    m_proton = 1.67262192e-24  # grams
    k_boltzmann = 1.38066e-16  # erg/K
    u = gas_file['InternalEnergy'][()]
    elec_ab = gas_file['ElectronAbundance'][()]
    hyd_ab = gas_file['GFM_Metals'][()][:, 0]
    mean_molec_weight = 4 * m_proton / (
        1 + 3 * hyd_ab + 4 * hyd_ab * elec_ab)
    temperature = (gamma - 1) * u / k_boltzmann * mean_molec_weight
    temperature *= 1e10
    return temperature


# Mr Krtxo \(ﾟ▽ﾟ)/
