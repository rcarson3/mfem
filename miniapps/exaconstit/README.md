# ExaConstit App for AM Constitutive Properties Applications
## Author:
Steven R. Wopschall

wopschall1@llnl.gov

Robert A. Carson

carson16@llnl.gov

Date: Aug. 6, 2017

Updated: Feb. 4, 2019

# Description: 
The purpose of this code app is to determine bulk constitutive properties of metals. This is a nonlinear quasi-static, implicit solid mechanics code built on the MFEM library based on an updated Lagrangian formulation (velocity based).
               
Currently, only Dirichlet boundary conditions (homogeneous and inhomogeneous by dof component) have been implemented. Neumann (traction) boundary conditions and a body force are not implemented. A new ExaModel class allows one to implement arbitrary constitutive models. The code currently successfully allows for various UMATs to be interfaced within the code framework. Development work is currently focused on allowing for the mechanical models to run on GPGPUs. 

The code supports either constant time steps or user supplied delta time steps. Boundary conditions are supplied for the velocity field applied on a surface. It supports a number of different preconditioned Krylov iterative solvers (PCG, GMRES, MINRES) for either symmetric or nonsymmetric positive-definite systems. 


## Remark:
See the included options.toml to see all of the various different options that are allowable in this code and their default values.

A TOML parser has been included within this directory, since it has an MIT license. The repository for it can be found at: https://github.com/skystrife/cpptoml .

Example UMATs maybe obtained from https://web.njit.edu/~sac3/Software.html . We have not included them due to a question of licensing. The ones that have been run and are known to work are the linear elasticity model and the neo-Hookean material. Although, we might be able to provide an example interface so users can base their interface/build scripts off of what's known to work.

Note: the grain.txt, props.txt and state.txt files are expected inputs for CP problems, specifically ones that use the Abaqus UMAT interface class under the ExaModel.

# Installing Notes:

* git clone the LLNL BLT library into the directory above MFEM. It can be obtained at https://github.com/LLNL/blt.git
* MFEM will need to be built with Conduit (built with HDF5). The easiest way to install Conduit is to use spack install instruction provided by Conduit.  

* Create a build directory and cd into there
* Run ```cmake .. -DENABLE_MPI=ON -DENABLE_FORTRAN=ON -DMFEM_DIR{mfem's installed cmake location} -DBLT_SOURCE_DIR=${BLT cloned location} -DECMECH_DIR=${ExaCMech installed cmake location} -DRAJA_DIR={RAJA installed location} -DSNLS_DIR={SNLS location in ExaCMech} -DMETIS_DIR={Metis used in mfem location} -DHYPRE_DIR={HYPRE install location} -DCONDUIT_DIR={Conduit install location} -DHDF5_ROOT:PATH={HDF5 install location}```
* Run ```make -j 4```


#  Future Implemenations Notes:
               
* Visco-plasticity constitutive model
* GPGPU material models
* A more in-depth README that better covers the different options available.
* debug ability to read different mesh formats
* An up-to-date example options.toml file
