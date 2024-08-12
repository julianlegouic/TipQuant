class PollenTube:

    """ Describes a pollen tube, attributes are meant to be set by the different modules. """

    def __init__(self):
        self.contour = None
        # spline parameters
        self.tck = None
        self.us = None
        # tube properties
        self.N = None
        self.T = None
        self.curvs = None
        self.displacements = None
        self.roi_indices = None
        # characterics
        self.tip_index = None
        self.direction = None
        # membrane
        self.membrane_indices = None

    @property
    def tip(self):
        return self.contour[self.tip_index]

    @property
    def membrane_contour(self):
        return self.contour[self.membrane_indices]

    @property
    def membrane_normals(self):
        return self.N[self.membrane_indices]

    @property
    def membrane_curvatures(self):
        return self.curvs[self.membrane_indices]
