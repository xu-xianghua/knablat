""" 
solver.py
solve conduction(heat conduction or diffusion) equation using FVM
"""
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
try:
    from sksparse.cholmod import cholesky
except:
    pass
from .mesh import Mesh, Cell, Face, Boundary

class Node:
    """ class Node """
    def __init__(self, C: float=1., T: float=300., id: int=None):
        self.T = T
        self.C = C
        self.id = None
        self.conductors = []
        self.source = 0.
        
    def initialize(self, T0: float) -> None:
        self.T = T0


class Conductor:
    """ class Conductor """
    def __init__(self, 
                 K: float, 
                 upstream: Node=None, 
                 downstream: Node=None) -> None:
        self.K = K
        self.upstream = upstream
        if self.upstream is not None:
            self.upstream.conductors.append(self) 
        self.downstream = downstream
        if self.downstream is not None:
            self.downstream.conductors.append(self)
    
    def opposite_node(self, node: Node) -> Node:
        """ return the opposite node """
        return self.downstream if node is self.upstream else self.upstream


class Solver:
    """ class Solver for solving conduction equation """
    def __init__(self):
        self.mesh = None
        self.inner_nodes = None
        self.boundary_nodes = None
        self.conductors = None
        
    def read_mesh(self, mesh_file: str):
        """ read mesh file to mesh """
        pass
            
    def set_boundary(self):
        """ set boundary conditions """
        pass
        
    def setup_network(self, rho: float, cp: float, k: float, T0: float) -> None:
        """ setup thermal network from mesh """
        self.mesh.set_Gamma(k)
        # create nodes
        self.inner_nodes = []
        for c in self.mesh.cells:
            nd = Node(c.volume*rho*cp, T0)
            c.node = nd
            self.inner_nodes.append(nd)
        self.boundary_nodes = []
        for b in self.mesh.boundary.values():
            if b.type == "constant":
                for fc in b.faces:
                    nd = Node(0., b.value)
                    fc.node = nd
                    self.boundary_nodes.append(nd)
            else:
                for fc in b.faces:
                    nd = Node(0., T0)
                    fc.node = nd
                    self.inner_nodes.append(nd)
        # create heat conductor
        self.conductors = []
        for fc in self.mesh.interior_faces:
            self.conductors.append(Conductor(fc.conductance*fc.Gamma, fc.owner.node, fc.neighbour.node))
            
        for b in self.mesh.boundary.values():
            for fc in b.faces:
                self.conductors.append(Conductor(fc.conductance*fc.Gamma, fc.owner.node, fc.node))
            if b.type == "fixedFlux":
                for fc in b.faces:
                    fc.node.source += b.value * np.linalg.norm(fc.normal)
            elif b.type == "convection":
                env_node = Node(0., b.value[1])
                self.boundary_nodes.append(env_node)
                for fc in b.faces:
                    self.conductors.append(Conductor(np.linalg.norm(fc.normal) * b.value[0], fc.node, env_node))

    def init_Jacobian(self) -> None:
        """ init Jacobian matrix """
        m = len(self.inner_nodes)
        I, J = [], []
        self.v_ind = {}            
        for i, n in enumerate(self.inner_nodes):
            n.id = i
        for j, n in enumerate(self.boundary_nodes, i+1):
            n.id = j
        for h in self.inner_nodes:
            I.append(h.id)
            J.append(h.id)
            for b in h.conductors:
                h2 = b.opposite_node(h)
                if h2.id < m and (h.id*m + h2.id) not in self.v_ind:
                    I.append(h.id)
                    J.append(h2.id)
        self.jacobian = sparse.csc_matrix(([-1]*len(I), (I, J)), shape=(m, m), dtype=float)
        a, b = self.jacobian.nonzero()
        for i in range(self.jacobian.nnz):
            self.v_ind[a[i]*m + b[i]] = i
        self.rhs = np.zeros(m)
        self.X = np.zeros(m + len(self.boundary_nodes))
        for c in self.boundary_nodes:
            self.X[c.id] = c.T
        self.Xinner = self.X[:m]

    def update_matrix_linear(self) -> None:
        """ update matrix data for direct solver: AT = b. 
        """
        m = len(self.inner_nodes)
        self.jacobian.data[:] = 0.
        self.rhs[:] = 0.
        for c in self.inner_nodes:
            for b in c.conductors:
                c2 = b.opposite_node(c)
                if c2.id < m:
                    self.jacobian.data[self.v_ind[c.id*m + c2.id]] = -b.K
                else:
                    self.rhs[c.id] += b.K*c2.T
                self.jacobian.data[self.v_ind[c.id*m + c.id]] += b.K
            self.rhs[c.id] += c.source 
        
    def update_result(self, X: np.ndarray=None) -> None:
        """ update result """
        if X is None:
            X = self.X
        for i, n in enumerate(self.inner_nodes):
            n.T = X[i]
            
    def green_gauss_cell_based_grad(self):
        """ calculate gradient of cell using green gauss method """
        for c in self.mesh.cells:
            c.grad.fill(0.)
        for fc in self.mesh.interior_faces:
            fv = self.X[fc.owner.node.id] * fc.gc + self.X[fc.neighbour.node.id] * (1 - fc.gc)
            flux_f = fv * fc.normal
            fc.owner.grad += flux_f
            fc.neighbour.grad -= flux_f
        for b in self.mesh.boundary.values():
            for fc in b.faces:
                fc.owner.grad += self.X[fc.node.id] * fc.normal
        for c in self.mesh.cells:
            c.grad /= c.volume
            
    def least_square_cell_based_grad(self):
        """ calculate gradient of cell using least square method """
        for c in self.mesh.cells:
            c.grad = c.G @ (self.X[c.n_id] - self.X[c.node.id])
                    
    def correct_cross_diffusion(self, rhs: np.ndarray, least_square: bool=True):
        """ correct cross diffusion, with over-relaxed approach """
        if least_square:
            self.least_square_cell_based_grad()
        else:
            self.green_gauss_cell_based_grad()
        for fc in self.mesh.interior_faces:
            fg = fc.owner.grad * fc.gc + fc.neighbour.grad * (1 - fc.gc)
            Dfc = fc.Gamma * fg @ fc.Tf
            rhs[fc.owner.node.id] += Dfc
            rhs[fc.neighbour.node.id] -= Dfc
        for b in self.mesh.boundary.values():
            for fc in b.faces:
                fc.grad = fc.owner.grad
                Dfc = fc.Gamma * fc.grad @ fc.Tf
                rhs[fc.owner.node.id] += Dfc
                if fc.node.id < len(rhs):
                    rhs[fc.node.id] -= Dfc
            
    def acct_heat_flux(self, bound: str, cross_diffusion: bool=False):
        """ calculate heat flux on boundary """
        Q = 0.
        for fc in self.mesh.boundary[bound].faces:
            Q += fc.conductance * fc.Gamma * (self.X[fc.owner.node.id] - fc.node.T)
            if cross_diffusion:
                Q -= fc.Gamma * fc.grad @ fc.Tf
        return Q
    
    def solve_linear(self, method: str='spsolve', use_umfpack=False) -> None:
        """ solve linear network AX=b, without cross diffusion correction """
        self.init_Jacobian()
        self.update_matrix_linear()
        if method == 'bicg':
            self.X, _ = splinalg.bicg(self.jacobian, self.rhs)
        elif method == 'cg':
            self.X, _ = splinalg.cg(self.jacobian, self.rhs)
        elif method == 'gmres':
            self.X, _ = splinalg.gmres(self.jacobian, self.rhs)
        elif method == 'minres':
            self.X, _ = splinalg.minres(self.jacobian, self.rhs)
        elif method == 'cholesky':
            self.X = cholesky(self.jacobian)(self.rhs)
        else:
            self.X = splinalg.spsolve(self.jacobian, self.rhs, use_umfpack=use_umfpack)
        self.update_result()
                
    def solve_unstructure(self, 
                          criterion: float=1e-5, 
                          use_cholesky: bool=False,
                          least_square: bool=False,
                          relax: float=0.9) -> bool:
        """ solve conduction equation with unstructure correct """
        err = 1.
        self.init_Jacobian()
        self.update_matrix_linear()
        if least_square:
            self.mesh.calc_lstsq_coef()
        if use_cholesky:
            factor = cholesky(self.jacobian)
        else:
            factor = splinalg.factorized(self.jacobian)
        self.Xinner[:] = factor(self.rhs)
        self.mesh.init_cell_grad()
        while err > criterion:
            rhs = self.rhs.copy()
            self.correct_cross_diffusion(rhs, least_square)
            oldX = self.X.copy()
            X = factor(rhs)
            self.Xinner[:] += relax*(X - self.Xinner)
            err = np.linalg.norm(self.X - oldX)
            print(f'err: {err}')
        
        self.update_result()
        return True

