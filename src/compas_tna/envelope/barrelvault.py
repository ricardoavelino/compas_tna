import math

from numpy import array
from numpy import ones
from numpy import zeros

from compas.datastructures import Mesh
from compas_tna.diagrams.diagram_rectangular import create_ortho_mesh
from compas_tna.envelope.parametricenvelope import ParametricEnvelope


def barrelvault_envelope(
    rise: float = 1.0,
    span: float = 10.0,
    x0: float = 0.0,
    y_span: tuple = (0.0, 10.0),
    thickness: float = 0.50,
    min_lb: float = 0.0,
    nx: int = 20,
    ny: int = 10,
):
    """Create an envelope for a barrel vault (semi-circular arch) geometry with given parameters.

    Parameters
    ----------
    rise : float, optional
        Height of the arch, by default 1.0
    span : float, optional
        Span of the arch, by default 10.0
    x0 : float, optional
        Starting coordinate of the arch, by default 0.0
    y_span : tuple, optional
        Span of the vault in y direction (perpendicular to arch), by default (0.0, 10.0)
    thickness : float, optional
        Thickness of the vault, by default 0.50
    min_lb : float, optional
        Parameter for lower bound in nodes in the boundary, by default 0.0
    nx : int, optional
        Number of vertices in x direction, by default 20
    ny : int, optional
        Number of vertices in y direction, by default 10

    Returns
    -------
    middle : Mesh
        Middle mesh
    intrados : Mesh
        Intrados mesh
    extrados : Mesh
        Extrados mesh
    """
    # Create base topology (rectangular mesh)
    x_span = (x0, x0 + span)
    base_topology = create_ortho_mesh(x_span=x_span, y_span=y_span, nx=nx, ny=ny)
    xyz0, faces_i = base_topology.to_vertices_and_faces()
    xi, yi, _ = array(xyz0).transpose()

    # Create middle surface
    zt = barrelvault_middle(xi, yi, rise, span, x0, min_lb)
    xyzt = array([xi, yi, zt.flatten()]).transpose()
    middle = Mesh.from_vertices_and_faces(xyzt, faces_i)
    middle.update_default_vertex_attributes(thickness=thickness)

    # Create upper and lower bounds
    zub, zlb = barrelvault_bounds(xi, yi, rise, span, x0, thickness, min_lb)
    xyzub = array([xi, yi, zub.flatten()]).transpose()
    xyzlb = array([xi, yi, zlb.flatten()]).transpose()

    extrados = Mesh.from_vertices_and_faces(xyzub, faces_i)
    intrados = Mesh.from_vertices_and_faces(xyzlb, faces_i)

    return intrados, extrados, middle


def barrelvault_middle(x, y, rise, span, x0, min_lb):
    """Compute middle of a barrel vault based on the parameters.

    Parameters
    ----------
    x : array
        x-coordinates of the points
    y : array
        y-coordinates of the points (not used, but kept for consistency)
    rise : float
        Height of the arch
    span : float
        Span of the arch
    x0 : float
        Starting coordinate of the arch
    min_lb : float
        Parameter for lower bound in nodes in the boundary

    Returns
    -------
    z : array
        Values of the middle surface in the points
    """
    # Calculate arch geometry
    radius = rise / 2 + (span**2 / (8 * rise))
    zc = radius - rise
    xc = span / 2 + x0

    z = zeros((len(x), 1))

    for i in range(len(x)):
        xi = x[i]
        # Clamp x to valid range
        if xi < x0:
            xi = x0
        elif xi > x0 + span:
            xi = x0 + span

        # Calculate z from circular arch equation
        radicand = radius**2 - (xi - xc) ** 2
        if radicand > 0:
            z[i] = math.sqrt(radicand) - zc
        else:
            z[i] = -min_lb

    return z


def barrelvault_bounds(x, y, rise, span, x0, thk, min_lb):
    """Compute upper and lower bounds of a barrel vault based on the parameters.

    Parameters
    ----------
    x : array
        x-coordinates of the points
    y : array
        y-coordinates of the points (not used, but kept for consistency)
    rise : float
        Height of the arch
    span : float
        Span of the arch
    x0 : float
        Starting coordinate of the arch
    thk : float
        Thickness of the vault
    min_lb : float
        Parameter for lower bound in nodes in the boundary

    Returns
    -------
    ub : array
        Values of the upper bound in the points
    lb : array
        Values of the lower bound in the points
    """
    # Calculate arch geometry
    radius = rise / 2 + (span**2 / (8 * rise))
    ri = radius - thk / 2  # intrados radius
    re = radius + thk / 2  # extrados radius
    zc = radius - rise
    xc = span / 2 + x0

    ub = ones((len(x), 1))
    lb = ones((len(x), 1)) * -min_lb

    for i in range(len(x)):
        xi = x[i]
        # Clamp x to valid range
        if xi < x0:
            xi = x0
        elif xi > x0 + span:
            xi = x0 + span

        # Upper bound (extrados)
        radicand_ub = re**2 - (xi - xc) ** 2
        if radicand_ub > 0:
            ub[i] = math.sqrt(radicand_ub) - zc
        else:
            ub[i] = -min_lb

        # Lower bound (intrados)
        radicand_lb = ri**2 - (xi - xc) ** 2
        if radicand_lb > 0:
            lb[i] = math.sqrt(radicand_lb) - zc
        # else: lb[i] already set to -min_lb

    return ub, lb


def barrelvault_bounds_derivatives(x, y, rise, span, x0, thk, min_lb):
    """Computes the sensitivities of upper and lower bounds in the x, y coordinates and thickness specified.

    Parameters
    ----------
    x : array
        x-coordinates of the points
    y : array
        y-coordinates of the points (not used, but kept for consistency)
    rise : float
        Height of the arch
    span : float
        Span of the arch
    x0 : float
        Starting coordinate of the arch
    thk : float
        Thickness of the vault
    min_lb : float
        Parameter for lower bound in nodes in the boundary

    Returns
    -------
    dub : array
        Values of the sensitivities for the upper bound in the points (dzub/dt)
    dlb : array
        Values of the sensitivities for the lower bound in the points (dzlb/dt)
    """
    # Calculate arch geometry
    radius = rise / 2 + (span**2 / (8 * rise))
    ri = radius - thk / 2
    re = radius + thk / 2
    zc = radius - rise
    xc = span / 2 + x0

    ub = ones((len(x), 1))
    lb = ones((len(x), 1)) * -min_lb
    dub = zeros((len(x), 1))
    dlb = zeros((len(x), 1))

    for i in range(len(x)):
        xi = x[i]
        # Clamp x to valid range
        if xi < x0:
            xi = x0
        elif xi > x0 + span:
            xi = x0 + span

        # Upper bound (extrados)
        radicand_ub = re**2 - (xi - xc) ** 2
        if radicand_ub > 0:
            ze = math.sqrt(radicand_ub) - zc
            ub[i] = ze
            dub[i] = re / (2 * ze)

        # Lower bound (intrados)
        radicand_lb = ri**2 - (xi - xc) ** 2
        if radicand_lb > 0:
            zi = math.sqrt(radicand_lb) - zc
            lb[i] = zi
            dlb[i] = -ri / (2 * zi)

    return dub, dlb


def barrelvault_bound_react(x, y, rise, span, x0, thk, min_lb, fixed):
    """Compute the bounds on the reaction vector of the barrel vault."""
    # TODO: Implement if needed
    pass


def barrelvault_bound_react_derivatives(x, y, rise, span, x0, thk, min_lb, fixed):
    """Compute the sensitivities of the bounds on the reaction vector of the barrel vault."""
    # TODO: Implement if needed
    pass


class BarrelVaultEnvelope(ParametricEnvelope):
    def __init__(
        self,
        rise: float = 1.0,
        span: float = 10.0,
        x0: float = 0.0,
        y_span: tuple = (0.0, 10.0),
        thickness: float = 0.50,
        min_lb: float = 0.0,
        nx: int = 20,
        ny: int = 10,
        **kwargs,
    ):
        super().__init__(thickness=thickness, **kwargs)
        self.rise = rise
        self.span = span
        self.x0 = x0
        self.y_span = y_span
        self.min_lb = min_lb
        self.nx = nx
        self.ny = ny

        self.update_envelope()  # Generate the intra/extra/middle meshes

    @property
    def __data__(self):
        data = super().__data__
        data["rise"] = self.rise
        data["span"] = self.span
        data["x0"] = self.x0
        data["y_span"] = self.y_span
        data["min_lb"] = self.min_lb
        data["nx"] = self.nx
        data["ny"] = self.ny
        return data

    def __str__(self):
        return f"BarrelVaultEnvelope(name={self.name})"

    def update_envelope(self):
        intrados, extrados, middle = barrelvault_envelope(
            rise=self.rise,
            span=self.span,
            x0=self.x0,
            y_span=self.y_span,
            thickness=self.thickness,
            min_lb=self.min_lb,
            nx=self.nx,
            ny=self.ny,
        )
        self.intrados = intrados
        self.extrados = extrados
        self.middle = middle

    def compute_middle(self, x, y):
        return barrelvault_middle(x, y, self.rise, self.span, self.x0, self.min_lb)

    def compute_bounds(self, x, y, thickness=None):
        if thickness is None:
            thickness = self.thickness
        else:
            self.thickness = thickness
        return barrelvault_bounds(x, y, self.rise, self.span, self.x0, thickness, self.min_lb)

    def compute_bounds_derivatives(self, x, y, thickness=None):
        if thickness is None:
            thickness = self.thickness
        else:
            self.thickness = thickness
        return barrelvault_bounds_derivatives(x, y, self.rise, self.span, self.x0, thickness, self.min_lb)

    def compute_bound_react(self, x, y, thickness=None, fixed=None):
        if thickness is None:
            thickness = self.thickness
        else:
            self.thickness = thickness
        return barrelvault_bound_react(x, y, self.rise, self.span, self.x0, thickness, self.min_lb, fixed)

    def compute_bound_react_derivatives(self, x, y, thickness=None, fixed=None):
        if thickness is None:
            thickness = self.thickness
        else:
            self.thickness = thickness
        return barrelvault_bound_react_derivatives(x, y, self.rise, self.span, self.x0, thickness, self.min_lb, fixed)
