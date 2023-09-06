""" 
mesh.py
classes for FVM mesh, unstructured grid
"""
from typing import List, Tuple, Dict
import numpy as np

class Face:
    """ class Face, described by vertices, 
    and all vertices are in the same plane """
    def __init__(self, vertices: List[np.ndarray], id: int=None):
        self.vertices = vertices
        self.id = id
        self.normal = None
        self.centroid = None
        self.owner = None
        self.neighbour = None
        self.value = None
        self.conductance = None
        self.node = None  # link to the corresponding node in thermal network
        
    def __repr__(self):
        """ face representation """
        return f'Face {self.id}: {self.vertices}'
    
    def setup(self):
        """ setup face, calculate normal and centroid """
        vn = len(self.vertices)
        if vn < 3:
            raise ValueError('Face must have at least 3 vertices')
        
        if vn == 3: # for triangle
            self.centroid = np.sum(self.vertices, axis=0) / vn
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            self.normal = np.cross(v1, v2) / 2.
        elif vn == 4: # for quadrilateral
            c1 = np.sum(self.vertices[:3], axis=0) / 3
            c2 = (self.vertices[0] + self.vertices[2] + self.vertices[3]) / 3.
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            n1 = np.cross(v1, v2) / 2.
            v3 = self.vertices[3] - self.vertices[0]
            n2 = np.cross(v2, v3) / 2.
            self.normal = n1 + n2
            self.centroid = (np.linalg.norm(n1) * c1 + np.linalg.norm(n2) * c2) / np.linalg.norm(self.normal)
        else: # for polygon
            total_area = 0.
            normal = np.zeros(3)
            centriod = np.zeros(3)
            geo_center = np.sum(self.vertices, axis=0) / vn           
            for i in range(vn):
                p1 = self.vertices[i]
                p2 = self.vertices[(i+1)%vn]
                v1 = p2 - p1
                v2 = geo_center - p1
                n12 = np.cross(v1, v2) / 2.
                area = np.linalg.norm(n12)
                total_area += area
                normal += n12
                centriod += area * np.sum([p1, p2, geo_center], axis=0) / 3.     
            self.centroid = centriod / total_area
            self.normal = normal
        
        
class Cell:
    """ class Cell, a cell is composed of faces """
    def __init__(self, faces: List[Face], id: int=None):
        self.faces = faces
        self.id = id
        self.centroid = None
        self.volume = None
        self.Gamma = 1. # diffusion coefficient, such as thermal conductivity
        self.node = None  # link to the corresponding node in thermal network
        
    def setup(self):
        """ setup cell, calculate centroid, volume """
        if self.faces is None or len(self.faces) < 3:
            raise ValueError('Cell must have at least 3 faces')
        total_volume = 0.
        centroid = np.zeros(3)
        geo_center = np.sum([fc.centroid for fc in self.faces], axis=0) / len(self.faces)
        for face in self.faces:
            fc = face.centroid
            # construct a pyramid with face as base and geo_center as top
            pyramid_centr = 0.75 * fc + 0.25 * geo_center
            pyramid_vol = abs((fc - geo_center) @ face.normal) / 3.
            total_volume += pyramid_vol
            centroid += pyramid_vol * pyramid_centr    
        self.volume = total_volume 
        if total_volume < 1e-32:
            raise ValueError(f'Cell{self.id} volume is too small')
        self.centroid = centroid / total_volume
        
    def calc_lstsq_coef(self, weight_correction: bool=False):
        """ calculate least square gradient coefficient matrxi """
        self.n_id = []  # neighbour cell/face id
        D = np.zeros((len(self.faces), 3))
        if weight_correction:
            W = np.zeros((len(self.faces), len(self.faces))) # weight matrix
        for i, face in enumerate(self.faces):
            if face.neighbour is None:
                p1 = face.centroid
                self.n_id.append(face.node.id)
            elif face.owner is self:
                p1 = face.neighbour.centroid
                self.n_id.append(face.neighbour.node.id)
            else:
                p1 = face.owner.centroid
                self.n_id.append(face.owner.node.id)
            di = p1 - self.centroid
            D[i,:] = di
            if weight_correction:
                W[i, i] = 1./np.linalg.norm(di)
        if weight_correction:
            self.G = np.linalg.inv(D.T @ W.T @ W @ D) @ D.T @ W.T @ W
        else:
            self.G = np.linalg.inv(D.T @ D) @ D.T
        
class CellBlock:
    """ class CellBlock, a cell block is composed of cells """
    def __init__(self, cells: List[Cell], id: int = None, label: str=None):
        self.cells = cells
        self.id = id
        self.label = label
        
    def setup(self):
        """ setup cell block """
        for cell in self.cells:
            cell.setup()
            
    def set_value(self, v: float):
        """ set cell value """
        for cell in self.cells:
            cell.value = v
            
    def set_Gamma(self, a: float):
        """ set diffusion coefficient """
        for cell in self.cells:
            cell.Gamma = a
                   
        
class Boundary:
    """ class Boundary, define boundary conditions """
    def __init__(self, bid: int, faces: List[Face], type: str, label: str=None):
        self.bid = bid
        self.faces = faces
        self.type = type
        self.label = label
        self.value = None
        
    def set_value(self, v: float):
        """ set boundary value """
        for face in self.faces:
            face.value = v
        
        
class Mesh:
    """ class Mesh, with vertices, faces, cells and boundary conditions """
    def __init__(self, 
                 vertices: np.ndarray=None, 
                 faces: List[Face]=None, 
                 cells: List[Cell]=None, 
                 blocks: List[CellBlock]=None,
                 boundary: Dict[str, Boundary]=None,
                 interior_faces: List[Face]=None):
        self.vertices = vertices
        self.faces = faces
        self.cells = cells
        self.blocks = blocks
        self.boundary = boundary
        self.interior_faces = interior_faces
        
    def setup(self):
        """ setup mesh """
        for fc in self.faces:
            p0 = fc.owner.centroid
            if fc.neighbour is not None:
                p1 = fc.neighbour.centroid
                gf = (fc.centroid - p0) @ fc.normal / ((p1 - p0) @ fc.normal)
            else:
                p1 = fc.centroid
                gf = 0.
            fc.gc = 1. - gf  # distance factor to owner cell
            fc.d_j = p1 - p0
            Ef = fc.normal @ fc.normal / (fc.d_j @ fc.normal) * fc.d_j  # face orthogonal vector, with over-relaxation
            # Ef = fc.normal @ fc.d_j / (fc.d_j @ fc.d_j) * fc.d_j  # face orthogonal vector, with minimum correction
            # Ef = np.linalg.norm(fc.normal) * fc.d_j / np.linalg.norm(fc.d_j)  # face orthogonal vector, with orthogonal correction
            fc.Tf = fc.normal - Ef  # face non-orthogonal vector
            fc.conductance = np.linalg.norm(Ef) / np.linalg.norm(fc.d_j)
            
    def calc_lstsq_coef(self):
        """ calculate least square gradient coefficient matrxi """
        for c in self.cells:
            c.calc_lstsq_coef()
            
    def init_cell_grad(self):
        """ initialize cell gradient """
        for cell in self.cells:
            cell.grad = np.zeros(3)
        for face in self.faces:
            face.grad = np.zeros(3)
                
    def set_Gamma(self, a: float):
        """ set diffusion coefficient """
        for cell in self.cells:
            cell.Gamma = a
        for fc in self.interior_faces:
            fc.Gamma = 1./(fc.gc/fc.owner.Gamma + (1. - fc.gc)/fc.neighbour.Gamma)
        for b in self.boundary.values():
            for fc in b.faces:
                fc.Gamma = fc.owner.Gamma
                    
    def total_volume(self):
        """ calculate total volume of mesh """
        return np.sum([cell.volume for cell in self.cells])
    
    def info(self):
        """ print mesh info """
        print(f'vertices number: {len(self.vertices)}')
        print(f'faces number: {len(self.faces)}')
        print(f'interior faces number: {len(self.interior_faces)}')
        print(f'cells number: {len(self.cells)}')
        print(f'boundary number: {len(self.boundary)}')
        print(f'total volume: {self.total_volume()}')
    
