import numpy as np

def generate_hermite_path(cone_x, cone_y, stop_dist=1.0, tangent_strength=None, num_points=50):
    """
    Calculates a smooth path to a cone in the car's local frame.
    - cone_x, cone_y: Position of the cone relative to the car.
    - stop_dist: Desired distance to stop in front of the cone.
    - tangent_strength: Controls 'curviness'. Default is 1.5x the distance.
    """
    # 1. Start: Car is at (0,0) facing +X
    p0 = np.array([0.0, 0.0])

    # 2. Goal: Find the point 'stop_dist' in front of the cone
    phi = np.arctan2(cone_y, cone_x) # Angle from car to cone
    p1 = np.array([
        cone_x - stop_dist * np.cos(phi),
        cone_y - stop_dist * np.sin(phi)
    ])

    # 3. Tangents: Direction car should be moving at start and end
    # Strength (L) affects how wide the turn is
    dist_to_goal = np.linalg.norm(p1 - p0)
    L = tangent_strength if tangent_strength else dist_to_goal * 1.5

    m0 = np.array([L, 0.0])              # Start tangent: straight forward
    m1 = np.array([L * np.cos(phi), L * np.sin(phi)]) # End tangent: facing cone

    # 4. Generate the Spline points
    t = np.linspace(0, 1, num_points)
    path = []
    for val in t:
        # Cubic Hermite Basis Functions
        h00 = 2*val**3 - 3*val**2 + 1
        h10 = val**3 - 2*val**2 + val
        h01 = -2*val**3 + 3*val**2
        h11 = val**3 - val**2

        # Calculate point on the curve
        point = h00*p0 + h10*m0 + h01*p1 + h11*m1
        path.append(point)

    return np.array(path)


path_points = generate_hermite_path(cone_x=4.0, cone_y=2.0, stop_dist=0.5)
