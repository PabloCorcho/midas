from numpy import array, zeros, interp, newaxis, sqrt, sum, cross, dot, sin, cos, arccos, linalg
from .smoothing_kernel import GaussianKernel
from . import cosmology

class Galaxy(object):
    """
    Def. This class represents a galaxy, containing different components such as stellar and gas particles.
    ···
    Attributes
    ----------
    name : str (optional, default='gal')
        Name of the galaxy to be used for storage.
    star_params : dict (optional)
        Input dictionary containing all stellar properties (position, vel, mass, metallicity)
    stars : dict #TODO
    gas_params : dict (optional)
        Dictionary containing all gas properties (position, vel, mass, metallicity, ...)
    gas : dict #TODO
    position : array (optional)
        (x, y, z) array vector corresponding to the position of the galaxy in the same frame as the particles.
        If not provided, the particles position vector will be assumed to be in the galaxy frame.
    velocity :
        (v_x, v_y, v_z) array vector corresponding to the velocity of the galaxy in the same frame as the particles.
        If not provided, all particles will be assumed to be in the galaxy rest frame.
    spin : array
        (L_x, L_y, L_z) array vector corresponding to the spin of the galaxy in the same frame as the particles.
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

    def __init__(self, kernel=None, **kwargs):

        # Interpolation kernel
        if kernel is None:
            self.kernel = GaussianKernel(mean=0, sigma=.3)
        else:
            self.kernel = kernel
        # Name of the galaxy
        self.name = kwargs.get('name', "gal")
        self.stars_params = kwargs.get('stars', None)
        self.build_stars()
        self.gas_params = kwargs.get('gas', None)
        self.build_gas()
        self.spin = kwargs.get('gal_spin', zeros(3))  # kpc/(km/s)
        self.velocity = kwargs.get('gal_vel', zeros(3))  # km/s
        self.position = kwargs.get('gal_pos', zeros(3))  # kpc

    def build_stars(self):
        """todo."""
        if self.stars_params is not None:
            for key in list(self.stars_params.keys()):
                self.stars[key] = self.stars_params[key][()]
        if 'ages' not in self.stars.keys():
            self.stars['ages'] = interp(
                self.stars['GFM_StellarFormationTime'],
                cosmology.scale_f[::-1], cosmology.age_f[::-1])

    def build_gas(self):
        """todo."""
        if self.gas_params is not None:
            for key in list(self.gas_params.keys()):
                self.gas[key] = self.gas_params[key][()]

    def set_to_galaxy_rest_frame(self):
        """Set the position and velocity of gas and stellar particles to the galaxy rest frame."""
        if self.stars_params is not None:
            self.stars['Velocities'] += -self.velocity[newaxis, :]
            self.stars['Coordinates'] += -self.position[newaxis, :]
        if self.gas_params is not None:
            self.gas['Velocities'] += -self.velocity[newaxis, :]
            self.gas['Coordinates'] += -self.position[newaxis, :]
        self.velocity = zeros(3)
        self.position = zeros(3)

    def proyect_galaxy(self, orthogonal_vector=None):
        """
        Compute the projected the positions and velocities of all galaxy elements.
        If no vector is provided, the particles will be projected to the XY plane.
        Arguments
        ---------
        orthogonal_vector: array (default=(0, 0, 1))
            Orthogonal vector to the projected plane.
        """
        if orthogonal_vector is None:
            if self.stars_params is not None:
                self.stars['ProjCoordinates'] = self.stars['Coordinates'].copy().T
                self.stars['ProjVelocities'] = self.stars['Velocities'].copy().T
            if self.gas_params is not None:
                self.gas['ProjCoordinates'] = self.gas['Coordinates'].copy().T
                self.gas['ProjVelocities'] = self.gas['Velocities'].copy().T
            self.proyection_vector = (None, None, None)
        else:
            norm = orthogonal_vector / sqrt(sum(orthogonal_vector**2))
            self.proyection_vector = norm
            b = cross(array([0, 0, 1]), norm)
            b /= sqrt(sum(b**2))
            theta = arccos(dot(array([0, 0, 1]), norm))
            q0, q1, q2, q3 = (cos(theta/2), sin(theta/2)*b[0],
                              sin(theta/2)*b[1], sin(theta/2)*b[2])
            # Quartenion matrix
            Q = zeros((3, 3))
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
            u, v, w = (dot(Q, array([1, 0, 0])),
                       dot(Q, array([0, 1, 0])),
                       dot(Q, array([0, 0, 1])))
            # Change of basis matrix
            W = array([u, v, w])
            inv_W = linalg.inv(W)
            if self.stars_params is not None:
                self.stars['ProjCoordinates'] = array([
                    sum(self.stars['Coordinates'] * u[newaxis, :], axis=1),
                    sum(self.stars['Coordinates'] * v[newaxis, :], axis=1),
                    sum(self.stars['Coordinates'] * w[newaxis, :], axis=1)
                    ])
                self.stars['ProjVelocities'] = array([
                    sum(self.stars['Velocities'] * inv_W[0, :][newaxis, :],
                        axis=1),
                    sum(self.stars['Velocities'] * inv_W[1, :][newaxis, :],
                        axis=1),
                    sum(self.stars['Velocities'] * inv_W[2, :][newaxis, :],
                        axis=1)
                    ])
            if self.gas_params is not None:
                self.gas['ProjCoordinates'] = array([
                    sum(self.gas['Coordinates'] * u[newaxis, :], axis=1),
                    sum(self.gas['Coordinates'] * v[newaxis, :], axis=1),
                    sum(self.gas['Coordinates'] * w[newaxis, :], axis=1)
                    ])
                self.gas['ProjVelocities'] = array([
                    sum(self.gas['Velocities'] * inv_W[0, :][newaxis, :],
                        axis=1),
                    sum(self.gas['Velocities'] * inv_W[1, :][newaxis, :],
                        axis=1),
                    sum(self.gas['Velocities'] * inv_W[2, :][newaxis, :],
                        axis=1)
                    ])
# Mr Krtxo \(ﾟ▽ﾟ)/
