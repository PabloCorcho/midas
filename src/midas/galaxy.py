from numpy import array, zeros, interp, newaxis, sqrt, sum, cross, dot, sin, cos, arccos, linalg
import cosmology

class Galaxy(object):
    """
    Def.

    This class represents a galaxy from IllustrisTNG simulations and contains
    different elements such as stellar and gas particles.
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', "")
        self.stars_params = kwargs.get('stars', None)
        self.build_stars()
        self.gas_params = kwargs.get('gas', None)
        self.build_gas()
        self.spin = kwargs.get('gal_spin', zeros(3))  # kpc/(km/s)
        self.velocity = kwargs.get('gal_vel', zeros(3))  # kpc/(km/s)
        self.position = kwargs.get('gal_pos', zeros(3))  # kpc/(km/s)

    def build_stars(self):
        """todo."""
        self.stars = {}
        if self.stars_params is not None:
            for key in list(self.stars_params.keys()):
                self.stars[key] = self.stars_params[key][()]
        if 'ages' not in self.stars.keys():
            self.stars['ages'] = interp(
                self.stars['GFM_StellarFormationTime'],
                cosmology.scale_f[::-1], cosmology.age_f[::-1])

    def build_gas(self):
        """todo."""
        self.gas = {}
        if self.gas_params is not None:
            for key in list(self.gas_params.keys()):
                self.gas[key] = self.gas_params[key][()]

    def set_to_galaxy_rest_frame(self):
        """todo."""
        if self.stars_params is not None:
            self.stars['Velocities'] += -self.velocity[newaxis, :]
            self.stars['Coordinates'] += -self.position[newaxis, :]
        if self.gas_params is not None:
            self.gas['Velocities'] += -self.velocity[newaxis, :]
            self.gas['Coordinates'] += -self.position[newaxis, :]
        self.velocity = zeros(3)
        self.position = zeros(3)

    def proyect_galaxy(self, orthogonal_vector):
        """todo."""
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