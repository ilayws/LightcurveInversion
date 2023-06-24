import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as sp
import pywavefront

N = 30 # Number of verticies
r1 = 10 # Radius of first sphere
r2 = 1 # Minimum radius of "second sphere"

def points_on_sphere(N, r):

    points = np.zeros((N,3))

    # Get angles
    psi = np.random.random(N)*2.0*np.pi
    theta = np.arccos(1.0-2.0*np.random.random(N))

    # Get coordinates
    points[:,0] = r*np.sin(theta)*np.cos(psi)
    points[:,1] = r*np.sin(theta)*np.sin(psi)
    points[:,2] = r*np.cos(theta)

    return points

def generate_random_sphere(N, r):
    # Get verticies
    verts = points_on_sphere(N, r)

    # Create convex hull
    hull = sp.ConvexHull(verts)

    # Make sure normal vector points inward
    fixed_simplices = []
    normals = []
    centroids = []
    for simplex in hull.simplices:
        # Calculate normal and centroid
        centroid = np.sum(verts[simplex,:], axis=0) / 3.0
        normal = np.cross(verts[simplex[1],:] - verts[simplex[0],:], verts[simplex[2],:] - verts[simplex[1],:])
        centroids.append(centroid)
        # Check the normal points outward, if it doesnt change order of verticies
        if np.dot(centroid, normal) > 0.0:
            fixed_simplices.append(simplex)
            normals.append(normal)
        else:
            fixed_simplices.append(simplex[::-1])
            normals.append(normal*(-1))

    return fixed_simplices,verts,normals,centroids


def move_point_on_sphere(point, r1, r2):
    # Moves point from surface of sphere radius r1 to surface of sphere radius r2
    return point*(r2/r1)

def nonconvex_shape(N, r1, r2):
    # Gets random point spherical mesh with a radius of r1
    simplices,verts,normals,centroids = generate_random_sphere(N, r1)
    simplices = np.array(simplices)
    new_verts = np.zeros((N,3)) # Array of new vertices (start of as 0)
    new_normals = []
    new_centroids = []
    for i in range(N):
        new_verts[i,:] = move_point_on_sphere(np.array([verts[i,0],verts[i,1],verts[i,2]]),r1,(np.random.randint(50,100,1)/50) + r2)
    # Moves each vertice from surface of r1 sphere to a sphere of a random radius
    return new_verts,simplices

def createobj(i, convex):
    if convex:
        faces,vertices,normals,centroids = generate_random_sphere(N, r1)
        name = "convexmesh" + str(i) +".obj"
        # Save the cube as a .obj file
        with open(name, 'w') as f:
            for v in vertices:
                f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
            for face in faces:
                f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))
        # Load the .obj file using the pywavefront library
        mesh = pywavefront.Wavefront(name)
    else:
        vertices, faces = nonconvex_shape(N, r1, r2)
        name = "nonconvexmesh" + str(i) +".obj"
        # Save the cube as a .obj file
        with open(name, 'w') as f:
            for v in vertices:
                f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
            for face in faces:
                f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))

        # Load the .obj file using the pywavefront library
        mesh = pywavefront.Wavefront(name)

files = 10 # Amount of files saved
convex = False
for i in range(files):
    createobj(i, convex=convex)
