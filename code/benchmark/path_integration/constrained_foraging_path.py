import numpy as np
import math
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
from abc import ABC, abstractmethod
from project.boolean_reservoir.code.utils.utils import make_rng
class Agent:
    """Holds all mutable agent state for a walk: position, velocity, and accelerations."""
    def __init__(self, dim, position=None):
        self.origin = np.zeros(dim) if position is None else np.asarray(position, dtype=float).copy()
        self.position = np.zeros(dim) if position is None else np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(dim)
        self.a_external = np.zeros(dim)
        self.a_boundary = np.zeros(dim)
        self.a_net = np.zeros(dim)

    def reset(self, position=None):
        self.position[:] = self.origin if position is None else np.asarray(position, dtype=float).copy() 
        self.velocity[:] = 0
        self.a_external[:] = 0
        self.a_boundary[:] = 0
        self.a_net[:] = 0


class Boundary(ABC):
    def __init__(self, points):
        self.points = points

    @abstractmethod
    def is_inside(self, p: np.array):
        pass

    @abstractmethod
    def react(self, agent: 'Agent', p_end, v_end, t) -> 'Agent':
        """
        Called after tentative kinematics. Updates agent in-place and returns it.
        Reads agent.position for p_start.
        Writes corrected position to agent.position, reaction acceleration to agent.a_boundary,
        and final velocity (v_end + a_boundary * t) to agent.velocity.
        p_end: tentative position after kinematics (not yet committed).
        v_end: tentative velocity after kinematics (not yet committed).
        t: timestep.
        """
        pass

    def clip_to_boundary(self, p: np.array) -> np.array:
        """Return the nearest point on or inside the boundary to p.
        Used to correct sub-epsilon floating-point drift at the start of a walk.
        Hard boundaries must override this; soft boundaries never need it."""
        raise NotImplementedError(f'{self.__class__.__name__} does not implement clip_to_boundary')

    def get_points(self):
        return self.points

    def __str__(self):
        return f'{self.__class__.__name__}'

class NoBoundary(Boundary):
    def __init__(self):
        super().__init__([])

    def is_inside(self, p):
        return True

    def react(self, agent, p_end, v_end, t):
        agent.position = np.asarray(p_end, dtype=float)
        agent.a_boundary = np.zeros_like(p_end)
        agent.velocity = np.asarray(v_end, dtype=float)
        return agent

class IntervalBoundary(Boundary):
    '''Always 1D context'''
    def __init__(self, radius=0.5, center=None, boundary_tolerance=1e-10):
        self.radius = abs(0.5 if radius is None else radius)
        self.center = 0 if center is None else center
        self.points = (self.center - self.radius, self.center + self.radius)
        self.min = min(self.points)
        self.max = max(self.points)
        self.boundary_tolerance = boundary_tolerance
    
    @staticmethod
    def _scalar(p):
        return float(np.asarray(p).flat[0])

    def is_inside(self, p):
        p = self._scalar(p)
        return self.min <= p <= self.max

    def is_on_boundary(self, p):
        p = self._scalar(p)
        return abs(p - self.min) < self.boundary_tolerance or abs(p - self.max) < self.boundary_tolerance

    def clip_to_boundary(self, p: np.array) -> np.array:
        return np.clip(np.asarray(p, dtype=float), self.min, self.max)

    def react(self, agent, p_end, v_end, t):
        p_scalar = self._scalar(p_end)
        v_scalar = self._scalar(v_end)
        if self.is_inside(p_scalar):
            agent.position = np.array([p_scalar])
            agent.a_boundary = np.zeros(1)
            agent.velocity = np.array([v_scalar])
            return agent
        # Elastic reflection: fold overshoot, flip velocity
        if p_scalar < self.min:
            p_reflected = 2 * self.min - p_scalar
            v_reflected = abs(v_scalar)   # bounce inward (positive direction)
        else:
            p_reflected = 2 * self.max - p_scalar
            v_reflected = -abs(v_scalar)  # bounce inward (negative direction)
        p_reflected = np.clip(p_reflected, self.min, self.max)
        a_reaction = np.array([(v_reflected - v_scalar) / t])
        agent.position = np.array([p_reflected])
        agent.a_boundary = a_reaction
        agent.velocity = np.array([v_reflected])
        return agent

class PolygonBoundary(Boundary):
    def __init__(self, points, boundary_tolerance=1e-10):
        super().__init__(points)
        self.polygon = Polygon(points)
        self.boundary = self.polygon.boundary
        self.boundary_tolerance = boundary_tolerance
    
    def get_points(self):
        return self.polygon.exterior.coords

    def is_inside(self, p):
        """Check if point is inside polygon (including boundary)"""
        point = Point(p)
        return self.polygon.contains(point) or self.polygon.touches(point)

    def _edge_inward_normal(self, intersection):
        """Return the inward unit normal of the polygon edge nearest to intersection."""
        coords = np.array(self.polygon.exterior.coords)
        centroid = np.array(self.polygon.centroid.coords[0])
        best_normal, best_dist = None, float('inf')
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len == 0:
                continue
            edge_dir = edge / edge_len
            t = np.clip(np.dot(intersection - p1, edge_dir), 0, edge_len)
            closest = p1 + t * edge_dir
            dist = np.linalg.norm(intersection - closest)
            if dist < best_dist:
                best_dist = dist
                normal = np.array([-edge_dir[1], edge_dir[0]])
                # Ensure normal points inward
                if np.dot(normal, centroid - p1) < 0:
                    normal = -normal
                best_normal = normal
        return best_normal

    def clip_to_boundary(self, p: np.array) -> np.array:
        nearest = self.polygon.boundary.interpolate(
            self.polygon.boundary.project(Point(p))
        )
        return np.array(nearest.coords[0])

    def react(self, agent, p_end, v_end, t):
        p_start = agent.position
        p_end = np.asarray(p_end)
        v_end = np.asarray(v_end)
        if self.is_inside(p_end):
            agent.position = p_end
            agent.a_boundary = np.zeros_like(p_end)
            agent.velocity = v_end
            return agent

        intersections = self.boundary.intersection(LineString([p_start, p_end]))
        if intersections.is_empty:
            agent.position = p_start.copy()
            agent.a_boundary = np.zeros_like(p_end)
            agent.velocity = v_end
            return agent

        _, closest = nearest_points(Point(p_start), intersections)
        intersection = np.array(closest.coords[0])

        # Ensure intersection is numerically on the boundary
        if not self.is_inside(intersection):
            boundary_point = self.polygon.boundary.interpolate(
                self.polygon.boundary.project(Point(intersection))
            )
            intersection = np.array(boundary_point.coords[0])

        # Reflect remaining displacement off the boundary edge
        normal = self._edge_inward_normal(intersection)
        remaining = p_end - intersection
        reflected = remaining - 2 * np.dot(remaining, normal) * normal
        result = intersection + reflected
        if not self.is_inside(result):
            result = intersection  # fallback

        # Elastic velocity reflection: flip normal component, keep tangential
        v_n = np.dot(v_end, normal) * normal
        v_reflected = v_end - 2 * v_n
        a_reaction = (v_reflected - v_end) / t
        agent.position = result
        agent.a_boundary = a_reaction
        agent.velocity = v_reflected
        return agent

class SoftBoundary(Boundary):
    """Soft boundary: smooth pull toward center, no hard wall. Always is_inside=True.
    Subclasses override _force_magnitude(distance) to implement different force laws."""
    def __init__(self, spring_constant=5.0, center=None):
        super().__init__([])
        self.spring_constant = spring_constant
        self._center = center  # deferred to first call if None (dim-agnostic)

    def _get_center(self, p):
        if self._center is None:
            return np.zeros_like(p)
        return np.asarray(self._center, dtype=float)

    def _force_magnitude(self, distance: float) -> float:
        """Override in subclasses. Returns force magnitude given distance from center."""
        raise NotImplementedError

    def is_inside(self, p):
        return True  # no hard wall — agent can drift anywhere

    def react(self, agent, p_end, v_end, t):
        p_end = np.asarray(p_end, dtype=float)
        center = self._get_center(p_end)
        displacement = p_end - center
        distance = np.linalg.norm(displacement)
        if distance == 0:
            a_boundary = np.zeros_like(p_end)
        else:
            magnitude = self._force_magnitude(distance)
            direction = -displacement / distance  # toward center
            a_boundary = self.spring_constant * magnitude * direction
        agent.position = p_end
        agent.a_boundary = a_boundary
        agent.velocity = v_end + a_boundary * t
        return agent

    def get_points(self):
        return []


class LinearSoftBoundary(SoftBoundary):
    """Force proportional to distance: F = k*d (Hookean spring toward center)."""
    def _force_magnitude(self, distance: float) -> float:
        return distance


class QuadraticSoftBoundary(SoftBoundary):
    """Force proportional to distance²: F = k*d²."""
    def _force_magnitude(self, distance: float) -> float:
        return distance ** 2


class CubicSoftBoundary(SoftBoundary):
    """Force proportional to distance³: F = k*d³."""
    def _force_magnitude(self, distance: float) -> float:
        return distance ** 3

class WalkStrategy(ABC):
    # Explicit stream for every stochastic strategy. Assign from the caller
    # (the dataset does: strategy.rng = make_rng(D.seed, 'generate')) so a walk
    # is reproducible from a seed rather than from ambient global RNG state.
    # Left None -> fresh entropy, so direct callers keep working unseeded.
    rng = None

    @property
    def _rng(self) -> np.random.Generator:
        if self.rng is None:
            self.rng = make_rng()
        return self.rng

    @abstractmethod
    def compute_step(self, agent: 'Agent', boundary: Boundary) -> 'Agent':
        """Updates agent state for one step and returns it."""
        pass

    def __str__(self):
        return f'{self.__class__.__name__}'

class AccelerationReplayStrategy(WalkStrategy):
    """Replays recorded external accelerations through full kinematics with no drag.
    The boundary reacts naturally to the replayed trajectory.
    For strategies with no friction, this trace is identical to the original walk.
    For strategies with friction, this trace shows what the walk would look like without drag."""
    def __init__(self, a_external, t=1.0):
        self._a_externals = iter(a_external)
        self.t = t

    def compute_step(self, agent, boundary):
        a_ext = next(self._a_externals)
        # No drag — replay external acceleration only
        new_p = agent.position + agent.velocity * self.t + 0.5 * a_ext * self.t ** 2
        v_end = agent.velocity + a_ext * self.t
        agent = boundary.react(agent, new_p, v_end, self.t)
        agent.a_external = a_ext
        agent.a_net = a_ext  # a_net = a_ext (no drag in replay)
        return agent

class PhysicsWalkStrategy(WalkStrategy):
    def __init__(self, max_acceleration=1.0, mass=1.0, friction_coeff=0.0, friction_order=2,
                 max_speed=np.inf, t=1.0, fail_in_place=True, max_attempts=100,
                 resolution_bits=None):
        self.max_acceleration = max_acceleration
        self.mass = mass
        self.friction_coeff = friction_coeff
        self.friction_order = friction_order  # 1: linear (honey), 2: quadratic (air)
        self.max_speed = max_speed
        self.t = t
        self.fail_in_place = fail_in_place
        self.max_attempts = max_attempts
        self.resolution_bits = resolution_bits

    def _sample_acceleration(self, shape):
        # 1. Random Direction: Gaussian noise gives a uniform sphere surface
        direction = self._rng.normal(0, 1, shape)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.zeros(shape)
        direction = direction / norm

        # 2. Random Magnitude: uniform in d-ball volume (u^(1/d) fills the ball uniformly)
        d = np.prod(shape)
        magnitude = self._rng.uniform(0, 1) ** (1.0 / d)

        return direction * magnitude * self.max_acceleration

    def _calculate_drag_acceleration(self, velocity):
        """Drag acceleration opposing movement: a_drag = F_drag / mass."""
        if self.friction_coeff == 0.0:
            return np.zeros_like(velocity)
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return np.zeros_like(velocity)
        if self.friction_order == 1:
            # Linear drag: F = -c*v → a = -(c/m)*v
            return -(self.friction_coeff / self.mass) * velocity
        elif self.friction_order == 2:
            # Quadratic drag: F = -c*|v|*v → a = -(c/m)*|v|*v
            return -(self.friction_coeff / self.mass) * speed * velocity
        else:
            raise ValueError("friction_order must be 1 (linear) or 2 (quadratic)")

    def compute_step(self, agent, boundary):
        for attempt in range(self.max_attempts):
            a_external = self._sample_acceleration(agent.position.shape)

            # --- UNIVERSAL QUANTIZATION BLOCK ---
            if self.resolution_bits is not None:
                levels = (2 ** self.resolution_bits) - 1

                # 1. Clip heavy tails to the maximum representable bounds (crucial for Levy flights)
                a_clipped = np.clip(a_external, -self.max_acceleration, self.max_acceleration)

                # 2. Normalize from [-max, +max] to [0.0, 1.0]
                a_norm = (a_clipped + self.max_acceleration) / (2 * self.max_acceleration)

                # 3. Scale to [0, levels - 1], round to nearest integer, and scale back to [0.0, 1.0]
                a_quantized = np.round(a_norm * (levels - 1)) / (levels - 1)

                # 4. Map back to the physical [-max, +max] range
                a_external = a_quantized * 2 * self.max_acceleration - self.max_acceleration
            # ------------------------------------

            a_drag = self._calculate_drag_acceleration(agent.velocity)
            a_net = a_external + a_drag

            # Kinematics: s = v*t + ½*a*t², new_v = v + a*t
            new_p = agent.position + agent.velocity * self.t + 0.5 * a_net * self.t ** 2
            v_end = agent.velocity + a_net * self.t

            speed = np.linalg.norm(v_end)
            if speed > self.max_speed:
                v_end = (v_end / speed) * self.max_speed

            if not np.isfinite(new_p).all() or not np.isfinite(v_end).all():
                continue

            agent = boundary.react(agent, new_p, v_end, self.t)
            agent.a_external = a_external
            agent.a_net = a_net
            return agent

        if self.fail_in_place:
            agent.reset(position=agent.position)
            return agent
        raise RuntimeError(f"Can't find a valid step after {self.max_attempts} attempts")

class SimpleRandomWalkStrategy(PhysicsWalkStrategy):
    """No friction, no nonlinearity. Momentum is linear so AccelerationReplayStrategy trace matches exactly."""
    def __init__(self, max_acceleration=1.0, mass=1.0, t=1.0, fail_in_place=True, max_attempts=100,
                 resolution_bits=None):
        super().__init__(
            max_acceleration=max_acceleration,
            mass=mass,
            friction_coeff=0.0,
            t=t,
            fail_in_place=fail_in_place,
            max_attempts=max_attempts,
            resolution_bits=resolution_bits,
        )

class LevyFlightStrategy(PhysicsWalkStrategy):
    '''alpha ≈ 1.5-2.0: Most animal foraging studies (albatrosses, deer, etc.)
    max_acceleration scales the Lévy distribution.
    '''
    def __init__(self, alpha=1.5, max_acceleration=0.1,
                 mass=1.0, friction_coeff=0.1, friction_order=2,
                 max_speed=np.inf, t=1.0, fail_in_place=True, max_attempts=100,
                 resolution_bits=None):
        super().__init__(
            max_acceleration=max_acceleration,
            mass=mass,
            friction_coeff=friction_coeff,
            friction_order=friction_order,
            max_speed=max_speed,
            t=t,
            fail_in_place=fail_in_place,
            max_attempts=max_attempts,
            resolution_bits=resolution_bits,
        )
        self.alpha = alpha

    def _sample_acceleration(self, shape):
        # Random direction (Gaussian → uniform sphere)
        direction = self._rng.normal(0, 1, shape)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.zeros(shape)
        direction = direction / norm
        # Lévy/Pareto heavy-tail magnitude
        magnitude = self._rng.pareto(self.alpha) * self.max_acceleration
        return direction * magnitude

def random_walk(dim: int, steps: int, strategy: WalkStrategy, boundary: Boundary, agent: Agent = None, origin=None, reset=True, rng=None):
    # rng: give the strategy an explicit stream (see WalkStrategy.rng). Without it
    # the strategy falls back to unseeded entropy.
    if rng is not None:
        strategy.rng = rng
    if agent is None:
        agent = Agent(dim, origin)
    elif reset:
        agent.reset()

    if not boundary.is_inside(agent.position):
        if (origin is not None or not reset) and not isinstance(boundary, SoftBoundary):
            # Floating-point drift: the previous walk's final position landed epsilon
            # outside the boundary edge. Clip to the nearest boundary point.
            # Being on the boundary is valid; drift never exceeds machine epsilon.
            agent.position = boundary.clip_to_boundary(agent.position)
        else:
            raise RuntimeError(f'Illegal start position according to {boundary}: {agent.position}')

    positions  = np.empty((steps + 1, dim))
    velocities = np.empty((steps, dim))
    a_exts     = np.empty((steps, dim))
    a_bnds     = np.empty((steps, dim))
    a_nets     = np.empty((steps, dim))

    positions[0] = agent.position
    for i in range(steps):
        agent = strategy.compute_step(agent, boundary)
        positions[i + 1] = agent.position
        velocities[i]    = agent.velocity
        a_exts[i]        = agent.a_external
        a_bnds[i]        = agent.a_boundary
        a_nets[i]        = agent.a_net

    return {
        'positions':  positions,   # (steps+1, dim)
        'velocities': velocities,  # (steps, dim)
        'a_ext':      a_exts,      # (steps, dim)
        'a_bnd':      a_bnds,      # (steps, dim)
        'a_net':      a_nets,      # (steps, dim)
        'agent':      agent,
    }

def compare_paths(path_a, path_b, suffix_a='real', suffix_b='ideal'):
    """Merge two random_walk dicts into a comparison dict with suffixed keys."""
    d = {f'{k}_{suffix_a}': v for k, v in path_a.items()}
    d.update({f'{k}_{suffix_b}': v for k, v in path_b.items()})
    return d

def extract_path(comparison, suffix='real'):
    """Extract a single random_walk dict from a comparison dict by suffix."""
    n = len(f'_{suffix}')
    return {k[:-n]: v for k, v in comparison.items() if k.endswith(f'_{suffix}')}

def generate_dual_trajectory(dim: int, steps: int, strategy: WalkStrategy, boundary: Boundary, origin=None):
    """Run two random walks with the same external accelerations.
    Returns a comparison dict with _real and _ideal suffixes.
    The real run uses the given strategy (may include drag/friction).
    The ideal run replays only the external accelerations with no drag.
    """
    actual = random_walk(dim, steps, strategy, boundary, origin=origin)
    t = getattr(strategy, 't', 1.0)
    replay = AccelerationReplayStrategy(actual['a_ext'], t=t)
    optimistic = random_walk(dim, steps, replay, boundary, origin=origin)
    return compare_paths(actual, optimistic)

def generate_polygon_points(n_sides, radius, rotation=0, center=None, decimals=10):
    """
    Generate points representing a regular polygon shape with n sides, the given radius, and rotation.
    The polygon will be centered around the specified center point.
    :param n_sides: Number of sides of the polygon
    :param radius: Radius of the polygon
    :param rotation: Rotation of the polygon in radians
    :param center: Center point of the polygon, defaults to (0, 0)
    :return: Numpy array of points representing the vertices of the polygon
    """
    if center is None: center = (0, 0)
    points = []
    angle_step = 2 * math.pi / n_sides
    for i in range(n_sides):
        # Calculate the base angle and apply rotation
        angle = i * angle_step + rotation
        
        # Generate point relative to center
        x = radius * math.cos(angle) + center[0]
        y = radius * math.sin(angle) + center[1]
        
        points.append((x, y))
    return np.round(points, decimals)

def stretch_polygon(boundary: PolygonBoundary, stretch_x: float=1, stretch_y: float=1):
    """
    Stretch the polygon points along the x and y axes.
    """
    stretched_points = [(x * stretch_x, y * stretch_y) for x, y in boundary.get_points()]
    boundary.points = stretched_points
    boundary.polygon = Polygon(stretched_points)
    return boundary

def to_polar(coords):
    """Convert cartesian coordinates to polar/spherical.
    coords: (..., D) where D is dimension
    """
    coords = np.asarray(coords)
    r = np.linalg.norm(coords, axis=-1, keepdims=True)
    
    if coords.shape[-1] == 1:
        return coords  # 1D, already scalar
    elif coords.shape[-1] == 2:
        theta = np.arctan2(coords[..., 1:2], coords[..., 0:1])
        return np.concatenate([r, theta], axis=-1)
    elif coords.shape[-1] == 3:
        theta = np.arctan2(coords[..., 1:2], coords[..., 0:1])
        phi = np.arccos(coords[..., 2:3] / (r + 1e-10))  # avoid div by zero
        return np.concatenate([r, theta, phi], axis=-1)
    else:
        raise ValueError(f"Unsupported dimension: {coords.shape[-1]}")
    
def to_cartesian(polar_coords):
    """Convert polar/spherical coordinates to cartesian.
    polar_coords: (..., D) where D is dimension
    """
    polar_coords = np.asarray(polar_coords)
    
    if polar_coords.shape[-1] == 1:
        return polar_coords  # 1D, already scalar
    elif polar_coords.shape[-1] == 2:
        r = polar_coords[..., 0:1]
        theta = polar_coords[..., 1:2]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.concatenate([x, y], axis=-1)
    elif polar_coords.shape[-1] == 3:
        r = polar_coords[..., 0:1]
        theta = polar_coords[..., 1:2]
        phi = polar_coords[..., 2:3]
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.concatenate([x, y, z], axis=-1)
    else:
        raise ValueError(f"Unsupported dimension: {polar_coords.shape[-1]}")




if __name__ == '__main__':
    from benchmark.path_integration.visualization import plot_random_walk
    # radial tollerance for accuracy formula: dx = 1/(2**b-1), dr = 2**.5 * dx 
    # dx := distance between points in grid in x and y direction (square grid)
    # b := resolution
    # dr := radial tollerance in a grid where sorounding points are within tollerance (1 point neighbourhood)

    # dim = 1
    # boundary = IntervalBoundary(radius=1.0)

    dim = 2
    octagon = generate_polygon_points(n_sides=8, radius=1, rotation=np.pi/4)
    boundary = PolygonBoundary(points=octagon)

    # strategy = LevyFlightStrategy()
    # strategy = SimpleRandomWalkStrategy()
    strategy = PhysicsWalkStrategy(max_acceleration=1.0, mass=1.0, friction_coeff=0.1, friction_order=2, max_speed=10)

    # steps = 100
    steps = 5
    d = generate_dual_trajectory(dim, steps, strategy, boundary)
    # plot_random_walk('/out/', d, strategy, boundary)
    plot_random_walk('/out/', extract_path(d, 'real'), strategy, boundary, overlays=['velocities'])