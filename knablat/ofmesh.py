""" 
ofmesh.py
read openfoam mesh files and setup pmesh
"""

from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
import os
import sys
from Ofpp import FoamMesh
from .mesh import Face, Cell, Mesh, Boundary

def read_of_mesh(path: str) -> Mesh:
    """ read openfoam mesh file """
    vertices = FoamMesh.parse_mesh_file(os.path.join(path, 'points'), FoamMesh.parse_points_content)
    faces_i = FoamMesh.parse_mesh_file(os.path.join(path, 'faces'), FoamMesh.parse_faces_content)
    owner = FoamMesh.parse_mesh_file(os.path.join(path, 'owner'), FoamMesh.parse_owner_neighbour_content)
    neighbour = FoamMesh.parse_mesh_file(os.path.join(path, 'neighbour'), FoamMesh.parse_owner_neighbour_content)
    faces = []
    for i, fc in enumerate(faces_i):
        face = Face([vertices[p] for p in fc], i)
        face.setup()
        faces.append(face)
    num_cells = max(max(owner), max(neighbour)) + 1
    cells = [Cell([], i) for i in range(num_cells)]
    for i, cn in enumerate(owner):
        cells[cn].faces.append(faces[i])
        faces[i].owner = cells[cn]
    for i, cn in enumerate(neighbour):
        cells[cn].faces.append(faces[i])
        faces[i].neighbour = cells[cn]
    for c in cells:
        c.setup()
        
    boundary = FoamMesh.parse_mesh_file(os.path.join(path, 'boundary'), FoamMesh.parse_boundary_content)
    boundary_dict = {}
    n = 0
    for k, v in boundary.items():
        # if k is bytes, convert to str
        if isinstance(k, bytes):
            k = k.decode('utf-8')
        boundary_dict[k] = Boundary(n, faces[v.start:v.start+v.num], v.type, k)
        n += 1
            
    interior_faces = faces[:len(neighbour)]
    blocks = []
    mesh = Mesh(vertices, faces, cells, blocks, boundary_dict, interior_faces)
    return mesh


if __name__ == '__main__':
    if len(sys.argv) < 2:
        path = '.'
    else:
        path = sys.argv[1]
    mesh = read_of_mesh(path)
    mesh.info()
    for k, b in mesh.boundary.items():
        print(k, b.type, len(b.faces))
    