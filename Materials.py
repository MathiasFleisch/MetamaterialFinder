# -*- coding: utf-8 -*-
"""
MetamaterialFinder

Material classes.

@author:
    Mathias Fleisch
    Polymer Competence Center Leoben GmbH
    mathias.fleisch@pccl.at
"""

class LinearElasticMaterial(object):
    def __init__(self, name, E, nu, density, G=None):
        """
        Linear elastic material

        Parameters
        ----------
        name : str
            Name of the material
        E : float
            Youngs modulus. Units: MPa
        nu : float
            Poisson's Ratio
        density : float
            Density of the material. Units: g/cm^3.
        G : float, optional
            Shear modulus, gets calculated if not given. Units: MPa.
            The default is None.

        Raises
        ------
        ValueError
            If the Poisson's Ratio is not in the range -1 <= nu <= 0.5.

        Returns
        -------
        None.

        """
        self.name = str(name) # To avoid problems with unicode and Abaqus
        self.E = E
        self.nu = nu
        if (self.nu < -1) | (self.nu > 0.5):
            raise ValueError("Poisson's Ratio not in range -1 <= nu <= 0.5")
        if not G:
            self.G = self._shear_modulus()
        else:
            self.G = G
        # Convert from g/cm^3 to (10^3 kg)/mm^3
        self.density = density*1e-9

    def _shear_modulus(self):
        return self.E/(2*(1 + self.nu))