"""
solver_test.py
test Solver
"""
import sys
import time
import numpy as np
from knablat.solver import Solver
from knablat.ofmesh import read_of_mesh

class TestSolver(Solver):
    """ class TestSolver """
    def __init__(self):
        super().__init__()
        
    def read_mesh(self, path: str):
        """ read mesh """
        self.mesh = read_of_mesh(path)
        self.mesh.setup()
        
    def set_boundary(self):
        """ set boundaries """
        self.mesh.boundary[b'bot_wall'].type = "constant"
        self.mesh.boundary[b'bot_wall'].value = 280.
        # self.mesh.boundary[b'bot_wall'].type = "convection"
        # self.mesh.boundary[b'bot_wall'].value = (1e4, 280.)
        self.mesh.boundary[b'hot_wall'].type = "fixedFlux"
        self.mesh.boundary[b'hot_wall'].value = 100000.
        # self.mesh.boundary[b'hot_wall'].type = "constant"
        # self.mesh.boundary[b'hot_wall'].value = 350.
        
    def show_result(self, cross_diffusion: bool=False):
        """ show result """
        T_hot = (sum([fc.node.T*np.linalg.norm(fc.normal) for fc in self.mesh.boundary[b'hot_wall'].faces]) 
                 / sum([np.linalg.norm(fc.normal) for fc in self.mesh.boundary[b'hot_wall'].faces]))
        print(f'mean T_hot: {T_hot}')
        T_node_mean = sum([nd.T*nd.C for nd in self.inner_nodes]) / sum([nd.C for nd in self.inner_nodes])
        print(f'mean T_node: {T_node_mean}')
        Qin = self.acct_heat_flux(b'hot_wall', cross_diffusion)
        print(f'Qin: {Qin}')
        Qout = self.acct_heat_flux(b'bot_wall', cross_diffusion)
        print(f'Qout: {Qout}')
        print(f'total volume: {self.mesh.total_volume()}')
      
      
def test_solver(path: str, direct=False, method='spsolve', least_square=False):
    """ test solver """  
    t0 = time.time()
    solver = TestSolver()
    solver.read_mesh(path)
    t1 = time.time()
    print(f'read mesh time: {t1-t0} s')
    solver.set_boundary()
    solver.setup_network(1., 1., 10., 300.)
    t2 = time.time()
    print(f'setup network time: {t2-t1} s')
    if direct:
        solver.solve_linear(method, use_umfpack=True)
    else:
        solver.solve_unstructure(criterion=1e-4, 
                                 use_cholesky=(method == 'cholesky'), 
                                 least_square=least_square,
                                 relax=0.99)
    t3 = time.time()
    print(f'method: {method}, time: {t3-t2} s')
    solver.show_result(not direct)
    print(f'total time: {t3-t0} s')
    
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python solver_test.py <mesh_path>')
        sys.exit()
    test_solver(sys.argv[1], False, least_square=True)

