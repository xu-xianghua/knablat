# Knablat

Knablat (K \nabla T) is a conduction/diffusion solver based on FVM(Finite Volume Method) using unstructured grid. It is developed for the purpose of learning or teaching the FVM.

## Features

### Mesh/grid

It uses unstructured mesh of 3D. The mesh can be tetrahedron, hexahedron, prism, pyramid, or mixed. 

The default mesh format is OpenFOAM mesh format. It can be converted from other mesh formats using tools by OpenFOAM, such as `fluentMeshToFoam` or `ideasUnvToFoam`.

And it is possible to write another mesh importer for other mesh formats. The default mesh importer is defined in ofmesh.py.

An ansys mesh(.msh) importer is provided in ansysmesh.py, which is modified from [meshio](https://github.com/nschloe/meshio).

### Boundary condition

It supports the following boundary conditions.

- Dirichlet
- von Neumann
- Robin or mixed

### Cross diffusion correction

For non-orothogonal mesh, the cross diffusion correction is needed. Two gradient schemes are supported, the green-gauss-cell-based gradient scheme and the least square gradient scheme. And the least square gradient scheme is recommended.

### Solver

It uses direct solver to solve the linear system of equations `AX=b`. The matrix A is stored in CSR sparse format(provided by scipy.sparse). The solver algrithm can be chosen from which is provided by scipy.sparse.linalg, such as CG, BiCGSTAB, GMRES, etc. If scikit-sparse is installed, the Cholesky decomposition can be used for better performance.

For unstructured mesh with cross diffusion correction, an iterative process is needed, and in each iteration, direct solver is used.

If the mesh is orthogonal, direct solver can be used without cross diffusion correction(Solver.solve_linear). And if the mesh is non-orothogonal, the cross diffusion correction is needed(use Solver.solve_unstructure).

## Usage/examples

### steps

1. Prepare mesh file in OpenFOAM format.
2. read mesh file and create mesh object.
3. setup boundary conditions.
4. setup network.
5. solve equations.

### setup boundary conditions

Boundary conditions are defined as a dictionary stored in `mesh.boundary`. The key is the name of boundary patch(defined in mesh file, for OpenFOAM mesh, it is defined in `polyMesh/boundary`), and the value is a Boundary object. The Boundary class has type, value, faces as attributes.

The types of boundary condition:

- `constant`: Dirichlet boundary condition with constant value. The value is the constant value.
- `fixedFlux`: von Neumann boundary condition with constant flux. The value is the constant flux.
- `convection`: Robin boundary condition with constant convection coefficient and constant temperature. The value is a tuple of (convection coefficient, temperature).

Demo code for setting boundary condition:

```python
    def set_boundary(self):
        """ set boundaries """
        self.mesh.boundary['bot_wall'].type = "constant"
        self.mesh.boundary['bot_wall'].value = 280.
        self.mesh.boundary['side_wall'].type = "convection"
        self.mesh.boundary['side_wall'].value = (1e4, 280.)
        self.mesh.boundary['hot_wall'].type = "fixedFlux"
        self.mesh.boundary['hot_wall'].value = 100000.
```

### define the specific solver

For a specific problem, the specific solver should be defined. The solver is a class inherited from Solver class. The specific solver class should implement the following methods:

- `read_mesh`: read mesh file and create mesh object.
- `set_boundary`: setup boundary conditions.
- others.

### example

The example is in the directory `examples/solver_test.py`. It is used to solve heat conduction for different boundary conditions.

The main code is as follows:

```python
import sys
import numpy as np
from knablat.solver import Solver
from knablat.ofmesh import read_of_mesh
from knablat.ansysmesh import read_ans_mesh

class TestSolver(Solver):
    """ class TestSolver """
    def __init__(self):
        super().__init__()
        
    def read_mesh(self, path: str):
        """ read mesh """
        if path.endswith('.msh'):
            self.mesh = read_ans_mesh(path)
        else:
            self.mesh = read_of_mesh(path)
        self.mesh.setup()
        
    def set_boundary(self):
        """ set boundaries """
        self.mesh.boundary['bot_wall'].type = "constant"
        self.mesh.boundary['bot_wall'].value = 280.
        self.mesh.boundary['hot_wall'].type = "fixedFlux"
        self.mesh.boundary['hot_wall'].value = 100000.
        
    def show_result(self, cross_diffusion: bool=False):
        """ show result """
        T_hot = (sum([fc.node.T*np.linalg.norm(fc.normal) for fc in self.mesh.boundary['hot_wall'].faces]) 
                 / sum([np.linalg.norm(fc.normal) for fc in self.mesh.boundary['hot_wall'].faces]))
        print(f'mean T_hot: {T_hot}')
        T_node_mean = sum([nd.T*nd.C for nd in self.inner_nodes]) / sum([nd.C for nd in self.inner_nodes])
        print(f'mean T_node: {T_node_mean}')
        Qin = self.acct_heat_flux('hot_wall', cross_diffusion)
        print(f'Qin: {Qin}')
        Qout = self.acct_heat_flux('bot_wall', cross_diffusion)
        print(f'Qout: {Qout}')
      
      
def test_solver(path: str, direct=False, method='spsolve', least_square=True):
    """ test solver """  
    solver = TestSolver()
    solver.read_mesh(path)
    solver.set_boundary()
    solver.setup_network(1., 1., 10., 300.)
    if direct:
        solver.solve_linear(method, use_umfpack=True)
    else:
        solver.solve_unstructure(criterion=1e-4, 
                                 use_cholesky=(method == 'cholesky'), 
                                 least_square=least_square,
                                 relax=0.99)
    solver.show_result(not direct)    
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python solver_test.py <mesh_path>')
        sys.exit()
    test_solver(sys.argv[1], False, least_square=True)
```

There are two demo mesh in the directory `examples/brick`, they have the same geometry of a brick with size of 10 cm long, 5 cm width and 2 cm height. The directory tetra is a 3D tetrahedron mesh with 5831 cells. the directory hexa is a 3D hexahedron mesh with 800 cells.

To run the example:

```bash
python solver_test.py brick/tetra
python solver_test.py brick/hexa
```

In the example, the top surface of the brick is heating by a fixed flux of 100000 W/m^2, the bottom surface is set to a constant temperature of 280 K, and the other surfaces are adiabatic. The temperature of top surface can be easily calculated theoretically as 480 K(the conductance is 10 W/m-K). The results of the solver using tetrahedron mesh are as follows:

- If solve the problem without cross diffusion correction, the temperature of top surface is 463.92 K, the error is 17 K.
- If solve the problem with cross diffusion correction, and choose the green-gauss-cell-based gradient scheme, the temperature of top surface is 467.58 K, the error is 12.4 K. 
- If choose the least square gradient scheme, the temperature of top surface is almost 480K, the error is less than 1e-6 K.

If the top and bottom surface are set to constant temperature of 350 K and 280 K, respectively, and the other surfaces are adiabatic, the heat flux of top surface can be easily calculated theoretically as 175 W/m^2.
The results of the solver using tetrahedron mesh are as follows:

- If solve the problem without cross diffusion correction, the heat flux of top surface is 190.87 W/m^2, the error is 15.87 W/m^2.
- If solve the problem with cross diffusion correction, and choose the green-gauss-cell-based gradient scheme, the heat flux of top surface is 186.66 W/m^2, the error is 11.66 W/m^2.
- If choose the least square gradient scheme, the heat flux of top surface is 175.0 W/m^2, the error is less than 1e-5 W/m^2.

And for hexahedron mesh, the results are the same as theorectical results no matter whether cross diffusion correction is used or not.


## Dependencies

- numpy
- scipy
- scikit-sparse(optional), for Cholesky decomposition
- Ofpp(optional), to import OpenFOAM mesh

