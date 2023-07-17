# -*- coding: utf-8 -*-
"""
MetamaterialFinder

Collection of pore functions.

Each pore functions inputs are defined as:
    pore_funct(x1, x2, ..., xn, rot_angle=0, x_offset=0, y_offset=0, n=100)

x1, x2, ..., xn: pore functions specific parameters (e.g. radius)
rot_angle: Rotation angle of pore (defined counter-clockwise)
x_offset: x-position of center of pore
y_offset: y-position of center of pore
n: Amount of points to create the curve

@author:
    Mathias Fleisch
    Polymer Competence Center Leoben GmbH
    mathias.fleisch@pccl.at
"""

import numpy as np


##################
# PORE FUNCTIONS #
##################

def circle(R, rot_angle=0, x_offset=0, y_offset=0, n=50):
    """
    Circular pore shape

    Parameters
    ----------
    R : float
        Radius of circle.
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 50.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    t = np.linspace(0, 2*np.pi, n)
    x = R*np.cos(t)
    y = R*np.sin(t)
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def squircle(R, S, rot_angle=0, x_offset=0, y_offset=0, n=1000):
    """
    Squircular pore shapes (smooth transformation from a circle to a square)
    Source: https://arxiv.org/abs/1604.02174

    Parameters
    ----------
    R : float
        "Radius".
    S : float
        "Squarness parameter", 0 <= S < 1
        S -> 0: circle
        S -> 1: square.
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 1000.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    t = np.linspace(0, 2*np.pi, n)
    x = R*np.cos(t)
    y = (R*np.sin(t))/(np.sqrt(1-S*(np.cos(t))**2))
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def sphylinder_projection(R, H, S, rot_angle=0, x_offset=0, y_offset=0, n=200):
    """
    Sphylinder pore shape, projected onto the plane (smooth transformation from
    an ellipse to a rectangle)
    Source: https://arxiv.org/abs/1604.02174

    Parameters
    ----------
    R : float
        Horizontal expansion (half).
    H : float
        Vertical expansion (half).
    S : float
        "Squarness parameter", 0 <= S < 1
        S -> 0: ellipse
        S -> 1: rectangle.
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 200.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    t = np.linspace(0, 2*np.pi, n)
    x = R*np.cos(t)
    y = (H*np.sin(t))/(np.sqrt(1. - S*(np.cos(t))**2))
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def superellipse(A, B, N, rot_angle=0, x_offset=0, y_offset=0, n=300):
    """
    Superellipse (or Lame curve) pore shape
    Source: https://en.wikipedia.org/wiki/Superellipse

    Parameters
    ----------
    A : float
        Horizontal expansion (half).
    B : float
        Vertical expansion (half).
    N : Shape of the superellipse ("Roundness")
        0 < N < 1: concave
        N = 1: rhombus
        1 < N < 2: convex rhombus
        N = 2: Ellipse
        N > 2: Rectangle with rounded corners
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 150.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    t = np.linspace(0, 2*np.pi, n)
    x = (np.abs(np.cos(t))**(2/N))*A*np.sign(np.cos(t))
    y = (np.abs(np.sin(t))**(2/N))*B*np.sign(np.sin(t))
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def superellipse_exp(A, B, N, M, rot_angle=0, x_offset=0, y_offset=0, n=300):
    """
    Superellipse (or Lame curve) pore shape with two exponents
    Source: https://en.wikipedia.org/wiki/Superellipse

    Parameters
    ----------
    A : float
        Horizontal expansion (half).
    B : float
        Vertical expansion (half).
    N : Shape of the superellipse in y-direction ("Roundness")
        0 < N < 1: concave
        N = 1: rhombus
        1 < N < 2: convex rhombus
        N = 2: Ellipse
        N > 2: Rectangle with rounded corners
    M : Shape of the superellipse in x-direction ("Roundness")
        0 < N < 1: concave
        N = 1: rhombus
        1 < N < 2: convex rhombus
        N = 2: Ellipse
        N > 2: Rectangle with rounded corners
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 150.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    t = np.linspace(0, 2*np.pi, n)
    x = (np.abs(np.cos(t))**(2/N))*A*np.sign(np.cos(t))
    y = (np.abs(np.sin(t))**(2/M))*B*np.sign(np.sin(t))
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def hippopede(a, b, rot_angle=0, x_offset=0, y_offset=0, n=100):
    """
    Hippopede pore shape
    Source: https://en.wikipedia.org/wiki/Hippopede

    a > b (to ensure one single pore)

    Parameters
    ----------
    a : float
        Vertical expansion (half).
    b : float
        Defines the constriction.
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 100.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    t = np.linspace(0, 2*np.pi, n)
    r_q = 4*b*(a - b*(np.sin(t))**2)
    r = np.sqrt(r_q)
    x = r*np.cos(t)
    y = r*np.sin(t)
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def n_sided_symmetric(c, r, N, d, rot_angle=0, x_offset=0, y_offset=0, n=300):
    """
    N-sided symmetric pore shape.
    Source: https://pubs.rsc.org/en/content/articlelanding/2016/RA/C6RA00295A

    Parameters
    ----------
    c : float
        Controls the porosity.
    r : float
        Controls the sharpness.
    N : int
        Number of folds.
    d : int
        +/- 1 orientation of the voids.
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 300.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    t = np.linspace(0, 2*np.pi, n)
    V = c*((1+r)-d*(-1)**((N+2)/2)*(r-1)*np.cos(N*t))
    x = V*np.cos(t)
    y = V*np.sin(t)
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def n_sided_polygon(r, N, rot_angle=0, x_offset=0, y_offset=0, n=1000):
    """
    N-sided polygon pore shape

    Parameters
    ----------
    r : float
        Diagonal of polynom (half).
    N : int
        Amount of sides.
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 1000.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    if N==4:
        rot_angle += 45
    t = np.linspace(0, 2*np.pi, n)
    pi = np.pi
    cos = np.cos
    sin = np.sin
    floor = np.floor
    x = r*cos(t)*(cos(pi/N))/(cos(2*(pi/N)*(t/(2*(pi/N))-floor(t/(2*(pi/N))))-pi/N))
    y = r*sin(t)*(cos(pi/N))/(cos(2*(pi/N)*(t/(2*(pi/N))-floor(t/(2*(pi/N))))-pi/N))
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def antichiral(R, L, T, rot_angle=0, x_offset=0, y_offset=0, n=30):
    """
    Piecewise pore to generate antichiral structure with 9 pores

    Parameters
    ----------
    R : float
        Radius of circular node.
    L : float
        Distance between circular nodes.
    T : float
        Thickness of connecting struts.
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 30.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    ao = np.arcsin(np.sqrt(2*R*T-T*T)/R)
    # First circle
    t1 = np.linspace((3*np.pi)/2.+ao, 2*np.pi, n)
    x1 = R*np.cos(t1)-L/2.
    y1 = R*np.sin(t1)-L/2.
    # Second circle
    t2 = np.linspace(0, np.pi/2.-ao, n)
    x2 = R*np.cos(t2)-L/2.
    y2 = R*np.sin(t2)+L/2.
    # Connection first - second
    x_12 = np.linspace(x1[-1], x2[0], n)[1:-1]
    y_12 = np.linspace(y1[-1], y2[0], n)[1:-1]
    # Third circle
    t3 = np.linspace(np.pi/2.+ao, np.pi, n)
    x3 = R*np.cos(t3)+L/2.
    y3 = R*np.sin(t3)+L/2.
    # Connection second - third
    x_23 = np.linspace(x2[-1], x3[0], n)[1:-1]
    y_23 = np.linspace(y2[-1], y3[0], n)[1:-1]
    # Fourth circle
    t4 = np.linspace(np.pi, (3*np.pi)/2.-ao, n)
    x4 = R*np.cos(t4)+L/2.
    y4 = R*np.sin(t4)-L/2.
    # Connection third - fourth
    x_34 = np.linspace(x3[-1], x4[0], n)[1:-1]
    y_34 = np.linspace(y3[-1], y4[0], n)[1:-1]
    # Connection fourth - first
    x_41 = np.linspace(x4[-1], x1[0])[1:]
    y_41 = np.linspace(y4[-1], y1[0])[1:]
    x = np.hstack((x1, x_12, x2, x_23, x3, x_34, x4, x_41))
    y = np.hstack((y1, y_12, y2, y_23, y3, y_34, y4, y_41))
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

def dumbbell(R, T, rot_angle=0, x_offset=0, y_offset=0, n=100):
    """
    Dumbbell pore shape

    Parameters
    ----------
    R : float
        Horizontal expansion (half).
    T : float
        Defines the constriction.
    rot_angle : float, optional
        Rotation angle of pore. The default is 0.
    x_offset : float, optional
        Offset of pore in x-direction. The default is 0.
    y_offset : float, optional
        Offset of pore in y-directino. The default is 0.
    n : int, optional
        Amount of points to create the curve. The default is 100.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape.
    """
    t = np.linspace(-1, 1, n)
    x1 = R*t
    y1 = R*(t**2 + T)*np.sqrt(1-t**2)
    x2 = R*t
    y2 = -R*(t**2 + T)*np.sqrt(1-t**2)
    x = np.hstack((x1, x2[::-1]))
    y = np.hstack((y1, y2[::-1]))
    return rot_offset(x, y, rot_angle, x_offset, y_offset)

#########
# UTILS #
#########

def rot_offset(x, y, rot_angle, x_offset, y_offset):
    """
    Rotates and offsets the pore.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates of the pore.
    y : numpy.ndarray
        y-coordinates of the pore.
    rot_angle : float, optional
        Rotation angle of pore.
    x_offset : float, optional
        Offset of pore in x-direction.
    y_offset : float, optional
        Offset of pore in y-directino.

    Returns
    -------
    numpy.ndarray
        Array with x-y coordinates of the pore shape (rotated and shifted).
    """
    if rot_angle != 0:
        x_rot, y_rot = rotate([x, y], rot_angle)
    else:
        x_rot, y_rot = x, y
    x_rot += x_offset
    y_rot += y_offset
    return np.vstack((x_rot, y_rot))

def rotate(coordinates, degrees=0):
    """
    Rotates the coordinates by the given angle in degrees, counterclockwise

    Parameters
    ----------
    coordinates : numpyp.ndarray
        x-y coordinates to be rotated.
    degrees : float, optional
        Angle of rotation. The default is 0.

    Returns
    -------
    numpy.ndarray
        Rotated array.

    """
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    return np.dot(R, coordinates)


if __name__ == '__main__':
    # Debug (to check implementation of pores)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        Lx = 20
        Ly = 20
        # points = circle(4, 0, 5, 5)
        # points = squircle(6.5, 0.9999, 0, 7.5, 7.5)
        # Antichiral
        points1 = sphylinder_projection(4.5, 2, 0.9999, 0, Lx/4., Ly/4.)
        points2 = sphylinder_projection(2, 4.5, 0.9999, 0, Lx/4, (3*Ly)/4.)
        points3 = sphylinder_projection(4.5, 2, 0.9999, 0, (3*Lx)/4, (3*Ly)/4.)
        points4 = sphylinder_projection(2, 4.5, 0.9999, 0, (3*Lx)/4, Ly/4.)
        points_list = [points1, points2, points3, points4]
        # points1 = superellipse(4, 3, 0.99, 0, Lx/2., Ly/2.)
        # points1 = n_sided_polygon(3.0, 6)
        # Antichiral
        # Lx = 50
        # Ly = 50
        points1 = antichiral(5, 25, 1, 90, Lx/2., Ly/2.)
        points2 = antichiral(5, 25, 1, 0, 0, Ly/2.)
        points3 = antichiral(5, 25, 1, 0, Lx, Ly/2.)
        points4 = antichiral(5, 25, 1, 90, 0, Ly)
        points5 = antichiral(5, 25, 1, 0, Lx/2., Ly)
        points6 = antichiral(5, 25, 1, 90, Lx, Ly)
        points7 = antichiral(5, 25, 1, 90, 0, 0)
        points8 = antichiral(5, 25, 1, 0, Lx/2., 0)
        points9 = antichiral(5, 25, 1, 90, Lx, 0)
        points_list = [points1, points2, points3, points4, points5, points6,
                        points7, points8, points9]
        # points = sphylinder_projection(3, 1, 0.8, 0)
        # points = superellipse(3, 1, 10)
        # points = hippopede(3, 1.9)
        Lx = 24
        Ly = 24
        # points = n_sided_symmetric(5, 0.3, 2, 1, 0, Lx/2., Ly/2.)
        # points = sphylinder_projection(9.5, 9.5, 0.999, 45, Lx/2., Ly/2.)
        # points = squircle(9, 0.999, 0, Lx/2., Ly/2.)
        # points = dumbbell(8, 0.9, 0, Lx/2., Ly/2.)
        # points = n_sided_polygon(8, 3, 22, Lx/2., Ly/2.)
        # points = superellipse(12, 5, 4, 0, Lx/2., Ly/2.)
        # points = superellipse_exp(5, 5, 0.8, 0.8, 0, Lx/2., Ly/2.)
        points1 = sphylinder_projection(10, 5, 0.5, 45, 0, 0)
        points2 = sphylinder_projection(10, 5, 0.5, 45, Lx, 0)
        points3 = sphylinder_projection(10, 5, 0.5, 135, Lx/2., Ly/2.)
        points4 = sphylinder_projection(10, 5, 0.5, 45, Lx, Ly)
        points5 = sphylinder_projection(10, 5, 0.5, 45, 0, Ly)
        # Superellipse
        # points1 = superellipse(10, 5, 3, 45, 0, 0)
        # points2 = superellipse(10, 5, 4, 45, Lx, 0)
        # points3 = superellipse(10, 5, 4, 135, Lx/2., Ly/2.)
        # points4 = superellipse(10, 5, 4, 45, Lx, Ly)
        # points5 = superellipse(10, 5, 4, 45, 0, Ly)
        # points_list = [points1, points2, points3, points4, points5]
        # N-sided-symmetric
        points1 = n_sided_symmetric(4.5, 0.7, 4, -1, 45, 0, 0)
        points2 = n_sided_symmetric(4.5, 0.7, 4, -1, 45, Lx, 0)
        points3 = n_sided_symmetric(4.5, 0.7, 4, -1, 0, Lx/2., Ly/2.)
        points4 = n_sided_symmetric(4.5, 0.7, 4, -1, 45, Lx, Ly)
        points5 = n_sided_symmetric(4.5, 0.7, 4, -1, 45, 0, Ly)
        points_list = [points1, points2, points3, points4, points5]
        # Hippopede
        a = 2.5
        b = 1
        points1 = hippopede(a, b, 0, 0, 0)
        points2 = hippopede(a, b, 90, Lx, 0)
        points3 = hippopede(a, b, 0, Lx, Ly)
        points4 = hippopede(a, b, 90, 0, Ly)
        # points_list = [points1, points2, points3, points4]

        r = 6.5
        N = 4
        points1 = n_sided_polygon(r, N, 40, Lx/4, Ly/4)
        points2 = n_sided_polygon(r, N, 50, 3*Lx/4, Ly/4)
        points3 = n_sided_polygon(r, N, 40, 3*Lx/4, 3*Ly/4)
        points4 = n_sided_polygon(r, N, 50, Lx/4, 3*Ly/4)
        # points_list = [points1, points2, points3, points4]

        r = 9.5
        T = 0.25
        points1 = dumbbell(r, T, 90, 0, 0)
        points2 = dumbbell(r, T, 90, Lx/2, Ly/2)
        # points_list = [points1, points2]
        
        fig, ax = plt.subplots(1, 1)
        for p in points_list:
            x, y = p
            ax.plot(x, y, marker='.')
            ax.fill(x, y, edgecolor=(0,0.5,0,1), facecolor=(0,0.5,0,0.4))
        ax.axis('off')
        rect = patches.Rectangle((0, 0), Lx, Ly, linewidth=1, edgecolor='black',
                                  linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlim((-Lx, 2*Lx))
        ax.set_ylim((-Lx, 2*Ly))
    except IndexError:
        pass
