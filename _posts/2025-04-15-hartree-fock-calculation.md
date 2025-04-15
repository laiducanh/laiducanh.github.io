---
title: Hartree-Fock calculation
categories: [Blogging, Tutorial]
tags: [quantum-chemistry, hartree-fock, water]
math: true
---

## Roothaan-Hall equation

The application of quantum mechanics in computational chemistry is represented by solving the Schrödinger equation, which leads to the determination of wavefunctions and eigen-energies from secular equations (reference to this [post](https://laiducanh.github.io/posts/energy-of-a-hydrogen-atom/)). Unfortunately, analytical solutions of the Schrödinger equation are only available for hydrogen-like systems that contain a single electron - for example, $H,H_2^+,HeH^{2+},He^+,Li^{2+}$, and so on. For more complex molecules, i.e. many-electron systems, approximations are required to estimate the energies and wavefunctions. Consequently, the core of quantum chemistry methods actually revolve how to get as highly accurate approximations as possible for a general chemical model. Current approaches focus on approximating the wavefunction starting from some fundamental physics. From this we can predict the total energy, electronic and magnetic properties, equilibrium nuclear geometries, nuclear dynamics, and more. Ab initio methods, such as Hartree-Fock, Moller-Plesset perturbation, Configuration Interactions, Coupled-cluster theory, make measurements of the wavefunction without experimental input. While we cannot solve exactly the molecular wavefunction for many-electron systems, it is true to say nobody knows what the wavefunctions look like. In fact, ab initio quantum chemistry methods use various mathematical expressions to represent the wavefunctions. In principle, any mathematical function can be used to construct molecular orbitals, such as polynomials, Fourier series, products, determinants, etc., but they need to be identical to the exact molecular wavefunction as much as possible. Another class of quantum chemistry methods, density functional theory, takes a different approach by focusing on electron density rather than the wavefunction itself. 

In the simplest quantum chemistry method, the Hartree-Fock method, the total molecular wavefunction $\Psi$ is approximated as a Slater determinant composed of occupied spin orbitals. To use these in practical calculations the molecular orbitals $\psi$ are expanded as a linear combination of basis functions $\phi$, which are based on the solution of the Schrödinger equation for the hydrogen atom (see this article). This linear combination of basis functions approach is commonly called a linear combination of atomic orbitals (LCAO) representation of the molecular orbitals

$$
\begin{equation}
\psi_i=\sum_{\mu}^{m}c_{\mu i}\phi_\mu
\end{equation}
$$

By expanding the spatial molecular orbital $\psi$’s in atomic orbital basis functions with corresponding coefficients $c$, the problem shifts from solving the wavefunction directly to finding the set of coefficients $c$ describing the molecular orbitals. In practice, we will always deal with basis functions, not molecular wavefunctions, meaning we work with basis function indices such as $\mu,\nu\,\rho,\sigma,\cdots$

The core process in the Hartree-Fock calculation is to solve Roothaan-Hall equation in the matrix form

$$
\begin{equation}
\mathbf{FC=SC\varepsilon}
\label{eq:roothaan-hall}
\end{equation}
$$

where $\mathbf F$ is the Fock matrix, $\mathbf C$ is the matrix of coefficients that determines the molecular orbitals, $\mathbf S$ is the overlap matrix, and $\varepsilon$ contains the eigen-energies of the molecular orbitals. Here $\mathbf F$, $\mathbf C$ and $\mathbf S$ would have to be $m\times m$ matrices, where $m$ is the number of basis functions. Since there are $m^2$ $F$’s, $c$’s and $S$’s, so ${\varepsilon}$ would be an $m\times m$ diagonal matrix with the nonzero elements $\varepsilon_1, \varepsilon_2,...,\varepsilon_m$. The central task in Hartree-Fock method is to build the Fock matrix, whose elements are computed as follows

$$
\begin{equation}
F_{\mu\nu}=h_{\mu\nu}+\sum_{\rho,\sigma}^m\sum_{j}^{d.o.}c_{\rho j}c_{\sigma j}\big(2g_{\mu\nu\rho\sigma}-g_{\mu\sigma\rho\nu} \big)
\label{eq:fock}
\end{equation}
$$

There are appearances of molecular integrals:

- Overlap integrals: $S_{\mu\nu}=\int_{-\infty}^{\infty}\phi_\mu^*(1)\phi_\nu(1) d\tau$
- Kinetic energy integrals: $T_{\mu\nu}=-\frac{1}{2}\int_{-\infty}^{\infty}\phi_\mu^*(1)\nabla_1^2\phi_\nu(1) d\tau$
- Nuclear attraction integrals: $V_{\mu\nu}=\int_{-\infty}^{\infty}\phi_\mu^*(1)\bigg(-\sum_{A}\frac{Z_A}{r_{1A}}\bigg)\phi_\nu(1) d\tau$
- Core Hamiltonian: $h_{\mu\nu}=T_{\mu\nu}+V_{\mu\nu}$
- Two-electron repulsion integrals: $g_{\mu\nu\rho\sigma}=\int_{-\infty}^{\infty}\phi_\mu^\*(1)\phi_\nu^\*(1)\frac{1}{r_{12}}\phi_\rho(2)\phi_\sigma(2)d\tau_1 d\tau_2$

For the purpose of this article, we will not discuss the evaluation of these integrals. We will assume that they have already computed and stored in corresponding data files below.

- [Overlap integrals](https://raw.githubusercontent.com/laiducanh/laiducanh.github.io/master/assets/data/intergrals/S.h2o.sto3g.npy)
- [Kinetic integrals](https://raw.githubusercontent.com/laiducanh/laiducanh.github.io/master/assets/data/intergrals/T.h2o.sto3g.npy)
- [Nuclear attraction integrals](https://raw.githubusercontent.com/laiducanh/laiducanh.github.io/master/assets/data/intergrals/V.h2o.sto3g.npy)
- [One-electron Hamiltonian](https://raw.githubusercontent.com/laiducanh/laiducanh.github.io/master/assets/data/intergrals/H.h2o.sto3g.npy)
- [Two-electron integrals](https://raw.githubusercontent.com/laiducanh/laiducanh.github.io/master/assets/data/intergrals/G.h2o.sto3g.npy)

## Implementation step by step

```python

import numpy as np
import math
from scipy import linalg

# Read integrals
S = np.load('S.h2o.sto3g.npy') # overlap integrals
T = np.load('T.h2o.sto3g.npy') # kinetic energy integrals
V = np.load('V.h2o.sto3g.npy') # nuclear attraction integrals
H = np.load('H.h2o.sto3g.npy') # (core) one-electron Hamiltonian
G = np.load('G.h2o.sto3g.npy') # two-electron integrals
```

In this post, we will try to solve the Hartree-Fock procedure for water molecule using a fixed geometry of the nuclei: O-H bond lengths of 0.95 Angstrom and a valence bond angle at oxygen of 104.5 degrees. 

$$
\begin{matrix}
O&0.0000000&0.0000000&0.1230031\\
H&0.0000000&-1.4194774&-0.9760738\\
H&0.0000000&1.4194774&-0.9760738
\end{matrix}
$$

We will initial the data for the molecule with lists of atoms, coordinates, and charges. Since we use the STO-3G basis set, there are 7 basis functions for the water molecule. 

```python

## dictionary: name --> atomic number
Z = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10}

atoms = ['O', 'H', 'H']
charges = [Z[atom] for atom in atoms]
coords = coords = np.array(
    [[0.0000000,  0.0000000,  0.1230031],
     [0.0000000, -1.4194774, -0.9760738],
     [0.0000000,  1.4194774, -0.9760738]]
)
nao = S.shape[0] # number of basis functions
nelec = 10 # number of electrons
```

The basis set composes of the following seven functions: basis function #1 is an oxygen $1s$ orbital, #2 is an oxygen $2s$ orbital, #3 is an oxygen $2p_x$ orbital, #4 is an oxygen $2p_y$ orbital, #5 is an oxygen $2p_z$ orbital, #6 is one hydrogen $1s$ orbital, and #7 is the other hydrogen $1s$ orbital. The overlap matrix is

$$
\mathbf{S}=\begin{bmatrix} \mathrm{O}\;1s & \mathrm{O}\;2s & \mathrm{O}\;2p_x & \mathrm{O}\;2p_y & \mathrm{O}\;2p_z & \mathrm{H}_a\;1s & \mathrm{H}_b\;1s & \\\ 1.000 & & & & & & &\mathrm{O}\;1s \\\ 0.237 & 1.000 & & & & & &\mathrm{O}\;2s \\\ 0.000 & 0.000 & 1.000 & & & & &\mathrm{O}\;2p_x \\\ 0.000 & 0.000 & 0.000 & 1.000 & & & &\mathrm{O}\;2p_y \\\ 0.000 & 0.000 & 0.000 & 0.000 & 1.000 & & &\mathrm{O}\;2p_z \\\ 0.055 & 0.480 & 0.000 & -0.313 & -0.242 & 1.000 & &\mathrm{H}_a\;1s \\\ 0.055 & 0.480 & 0.000 & 0.313 & -0.242 & 0.256 & 1.000&\mathrm{H}_b\;1s \end{bmatrix}
$$

There are many noteworthy features in $\mathbf S$. First, it is shown in a lower packed triangular form because every element $j,i$ is the same as the element $i,j$ by symmetry, and every diagonal element is exactly 1 because the basis functions are normalized. Note that, again by symmetry, within oxygen every $p$ orbital is orthogonal (zero overlap) with every $s$ orbital and with each other, but the two $s$ orbitals do overlap (this is due to the fact that they are not pure hydrogenic orbitals - which would indeed be orthogonal - but they have been optimized, so $S_{12}=0.237$). Note also that the oxygen $1s$ orbital overlaps about an order of magnitude less with any hydrogen $1s$ orbital than does the oxygen $2s$ orbital, reflecting how much more rapidly the first quantum-level orbital decays compared to the second. Note that by symmetry the oxygen $p_x$ cannot overlap with the hydrogen $1s$ functions (positive overlap below the plan exactly cancels negative overlap above the plane) and that the oxygen $p_y$ overlaps with the two hydrogen $1s$ orbitals equally in magnitude but with different sign because the $p$ orbital has different phase at its different ends. Finally, the overlap of the $p_z$ is identical with each H $1s$ because it is not changing which lobe it uses to interact. 

The kinetic energy matrix (in a.u.) is

$$
\mathbf{T}=  \left[\begin{matrix}\mathrm{O}\;1s & \mathrm{O}\;2s & \mathrm{O}\;2p_x & \mathrm{O}\;2p_y & \mathrm{O}\;2p_z & \mathrm{H}_a\;1s & \mathrm{H}_b\;1s & \\ 29.003 & & & & & & & O\;1s\\\ -0.168 & 0.808 & & & & & & O\;2s\\\ 0.000 & 0.000 & 2.529 & & & & & O\;2p_x\\\ 0.000 & 0.000 & 0.000 & 2.529 & & & & O\;2p_y\\\ 0.000 & 0.000 & 0.000 & 0.000 & 2.529 & & & O\;2p_z\\\ -0.002 & 0.132 & 0.000 & -0.229 & -0.178 & 0.760 & & H_a\:1s\\\ -0.002 & 0.132 & 0.000 & 0.229 & -0.178 & 0.009 & 0.760 &H_b\;1s \end{matrix}\right]
$$

Notice that every diagonal term is much larger than any off-diagonal term. Recall that each kinetic energy integral, involves the Laplacian operator, $\nabla^2$. The Laplacian reports back the sum of second derivatives in all coordinate directions. That is, it is a measure of how fast the slope of the function is changing in various directions. If we take two atomic orbitals $\mu$ and $\nu$ far apart from each other, then since atomic orbital functions go to zero at least exponentially fast with distance, $\nu$ is likely to be very flat where $\mu$ is large. The second derivative of a flat function is approximately zero. So, every point in the integration will be roughly the amplitude of $\mu$ times zero, and not much will accumulate. For the diagonal elements, on the other hand, the interesting second derivatives will occur where the function has maximum amplitude (amongst other places) so the accumulation should be much larger. Notice also that off-diagonal terms can be negative. That is because there is no real physical meaning to a kinetic energy expectation value involving two different orbitals. It is just an integral that appears in the complete secular determinant. Symmetry again keeps $p$ orbitals from mixing with $s$ orbitals or with each other. 

The nuclear attraction matrix is

$$
\mathbf{V}= \left[\begin{matrix}\mathrm{O}\;1s & \mathrm{O}\;2s & \mathrm{O}\;2p_x & \mathrm{O}\;2p_y & \mathrm{O}\;2p_z & \mathrm{H}_a\;1s & \mathrm{H}_b\;1s & \\ -61.733 & & & & & & &O\;1s\\\ -7.447 & -10.151 & & & & & &O\;2s\\\ 0.000 & 0.000 & -9.993 & & & & &O\;2p_x\\\ 0.000 & 0.000 & 0.000 & -10.152 & & & &O\;2p_y\\\ 0.019 & 0.226 & 0.000 & 0.000 & -10.088 & & &O\;2p_z\\\ -1.778 & -3.920 & 0.000 & 2.277 & 1.838 & -5.867 & &H_a\;1s\\\ -1.778 & -3.920 & 0.000 & -2.277 & 1.838 & -1.652 & -5.867&H_b\;1s\end{matrix} \right]
$$

Again, diagonal elements are bigger than off-diagonal elements because the $1/r$ operator acting on a basis function $\nu$ will ensure that the largest contribution to the overall integral will come from the nucleus $k$ on which basis function $\nu$ resides. Unless $\mu$ also has significant amplitude around that nucleus, it will multiply the result by roughly zero and the whole integral will be small. Again, positive values can arise when two different functions are involved even though electrons in a single orbital must always be attracted to nuclei and thus diagonal elements must always be negative. Note that the $p$ orbitals all have different nuclear attractions. That is because, although they all have the same attraction to the O nucleus, they have different amplitudes at the H nuclei. The $p_x$ orbital has zero amplitude at the H nuclei since they are in its nodal plane, so it has the smallest nuclear attraction integral. The $p_z$ orbital has somewhat smaller amplitude at the H nuclei than the $p_y$ orbital because the bond angle is greater than 90 degrees, i.e. 104.5 degrees. If it were 90 degrees the O-H bonds would bisect the $p_y$ and $p_z$ orbitals and their amplitudes at the H nuclei would necessarily be the same. Thus, the nuclear attraction integral for the latter orbital is slightly smaller than for the former. 

We define the density matrix whose elements

$$
\begin{equation}
D_{\rho\sigma}=2\sum_{j}^{d.o.}c_{\rho j}c_{\sigma j}
\end{equation}
$$

The Fock matrix from equation \eqref{eq:fock} can be modified as

$$
\begin{equation}
F_{\mu\nu}=h_{\mu\nu}+\sum_{\rho,\sigma}^mD_{\rho\sigma}\left(g_{\mu\nu\rho\sigma}-\frac{1}{2}g_{\mu\sigma\rho\nu} \right)
\label{eq:fock2}
\end{equation}
$$

In the end of a Hartree-Fock calculation, we want to find a set of molecular orbitals determined by coefficient matrix $\mathbf C$, and the eigen-energy matrix $\varepsilon$ of the molecular orbitals through the Roothan-Hall equations. However, each element of the Fock matrix $\mathbf F$ is calculated from the density matrix which is defined in terms of the coefficient matrix $\mathbf C$. It looks like we are faced with a dilemma: the point of calculating $\mathbf F$ is to get $\mathbf C$, but to get $\mathbf F$ we need $\mathbf C$. Let us try with a guess for density matrix, then solve the Roothaan-Hall equation to get a solution. In this case, we have no information on what the density may look like, but we know it should be a $m\times m$ matrix where $m$ is the number of basis functions. The simplest thing to do is simply to start out with a null density matrix: $\mathbf{D=0}$, which is so-called the core Hamiltonian guess. 

```python

def make_guess(nao:int):
    # create an empty naoxnao density matrix;
    D = np.zeros((nao,nao)) 
    return D
	
# The density matrix with an initial guess
D = make_guess(nao)
```

We are able to construct the first Fock matrix.

```python

def buildF(H, G, D):
    ## using the indices m and n in lieu of "mu" and "nu" ...
    ## ... initialize the naoxnao matrix P which is the two-electron contribution ...
    ## ... calculate the two-electron contribution ...
    ## ... by contracting the density matrix with the two-electron integrals
    nao = H.shape[0]
    P = np.zeros((nao,nao)) 
    for m in range(nao):
        for n in range(nao):
            P[m,n] = 0.0
            for r in range(nao):
                for s in range(nao):
                    P[m,n] += D[r,s] * (G[m,n,r,s] - 0.5*G[m,s,r,n])
    return H + P

# Calculate the Fock matrix
F = buildF(H, G, D)
```

The Roothaan-Hall matrix equations (equation \eqref{eq:roothaan-hall}) looks really similar to the eigenvalue formalism as [the Hamiltonian equation](https://laiducanh.github.io/posts/energy-of-a-hydrogen-atom/) for the hydrogen atom, except that now we have the overlap matrix on the right. As I said earlier, the overlap matrix is not identity because basis functions are non-orthogonal. But if we work with an orthogonal basis set, the overlap matrix becomes identity and the Roothaan-Hall equations turn to the standard eigenvalue equations so that we can diagonalize the Fock matrix to get the coefficients and the energy levels, just as we did for hydrogen atom. Our task now is to transform the basis functions $\phi$ to a new set that is orthogonal $\widehat\phi$, by a matrix $\mathbf X$

$$
\begin{equation}
\widehat{\phi}_\mu=\sum_i{X}_{\mu i}\phi_\mu
\end{equation}
$$

Since we transformed the basis function, the coefficients $\mathbf{C}$ need to be adjusted 

$$
\begin{equation}
{\mathbf{C}'}=\mathbf{X}^{-1}\mathbf{C}=\mathbf{S}^{1/2}\mathbf{C}
\end{equation}
$$

In this article, I follow the Lowdin orthogonalization process, also known as symmetric orthogonalization, where $\mathbf{X}=\mathbf{S}^{-1/2}$.

```python

# generate transformation matrix X
X = linalg.sqrtm(linalg.inv(S)) 
```

We can modify the Roothaan-Hall equations to use these orthogonal coefficients by using the resolution of the identity $\mathbf{I=S^{-1/2}S^{1/2}}$

$$
\begin{aligned}\mathbf{FC}&=\mathbf{SC\varepsilon}\\\mathbf{F}\mathbf{S}^{-1/2}\mathbf{S}^{1/2}\mathbf{C}&=\mathbf{S}\mathbf{S}^{-1/2}\mathbf{S}^{1/2}\mathbf{C}\mathbf{\varepsilon}\\
\mathbf{F}\mathbf{S}^{-1/2}{\mathbf{C}'}&=\mathbf{S}^{1/2}{\mathbf{C}'}\mathbf{\varepsilon}\\
\mathbf{S}^{-1/2}\mathbf{F}\mathbf{S}^{-1/2}{\mathbf{C}'}&=\mathbf{S}^{-1/2}\mathbf{S}^{1/2}{\mathbf{C}'}\mathbf{\varepsilon}\\
{\mathbf{F}'}{\mathbf{C}'}&={\mathbf{C}'}\mathbf{\varepsilon}
\end{aligned}
$$

The transformed the Fock matrix is then defined as 

$$
\begin{equation}
\mathbf{F'=S^{-1/2}FS^{-1/2}}
\end{equation}
$$

which can be diagonalized to give $\mathbf C'$ and $\varepsilon$. Transformation of $\mathbf{C'}$ to $\mathbf{C}$ gives the coefficients $c_{\mu i}$ in the expansion of the MO’s $\psi$ in terms of basis functions $\phi$. 

```python

# Solve the Roothaan-Hall equation
Fp = X @ F @ X
e, Cp = np.linalg.eigh(Fp)
C = X @ Cp
```

Let see how the density matrix $\mathbf D$ changes from the initial guess

```python

nelec = 10 # number of electrons 

# save the old density matrix for comparison
D_old = D.copy()

# form a new density matrix D from the molecular orbitals C
for m in range(nao):
    for n in range(nao):
        D[m,n] = 0.0
        for a in range(int(nelec/2)):
            D[m,n] += 2 * (C[m,a] * C[n,a])
```

Obviously, the new density matrix is not zero. Is this new density matrix better than our initial guess? If you plug the initial guess $\mathbf{D=0}$ into the equation \eqref{eq:fock2} the Fock matrix element is $F_{\mu\nu}=h_{\mu\nu}$ (that’s why it is called the core Hamiltonian guess). Physically, this means that the electrons have the correct kinetic energy and attraction with the nuclei, but do not interact with the other electrons at all. This is a much harsher approximation than the Hartree-Fock approximation, of course, and gives a very poor description of the molecule. Therefore, a non-zero density matrix does take some electron correlations into account. As a result, we have improved our density. But we will wonder is the new density matrix the best density we can have? I am not sure. So let us try with a new iteration, which we initialize the density matrix with the current density matrix. Technically, we will do iteratively several cycles until the solution looks “similar” to the guess. It means we cannot further improve the density from the guess, and we have met the convergence. In other words, the solution of the Roothaan-Hall equations must be self-consistent

$$
\mathbf{C}^{(k)} \rightarrow \mathbf{D}^{(k)} \rightarrow \mathbf{F}^{(k)} \rightarrow \mathbf{C}^{(k+1)} \xrightarrow{k\rightarrow\infty}\mathbf{C}^{(k)}
$$

As we continue the self-consistent Hartree-Fock procedure, we have to have some measures for how close to self-consistency (and hence the optimal Hartree-Fock energy) we are. In this implementation, I will use the maximum error (MAXE) that computes the largest difference between two density matrices in consecutive iterations.

```python

# Calculate convergence
conv = np.max(np.abs(D - D_old))
```

We will set the convergence threshold as `1e-8`. When the change in the density matrix is lessen than `1e-8`, we say the calculation is self-consistent. Once the calculation converged, the final task is to compute the total energy and the optimized coefficients that form the molecular orbitals. The electronic energy in Hartree-Fock method is defined as

$$
\begin{equation}
E_{el}=\frac{1}{2}\sum_{\mu,\nu}D_{\mu\nu}(h_{\mu\nu}+F_{\mu\nu})
\end{equation}
$$

Remember, we need to add the nuclear repulsion energy to get the total energy

$$
\begin{equation}
E_{total}=E_{el}+\sum_{B>A}^N  \frac{Z_AZ_B}{R_{AB}}
\end{equation}
$$

```python

def nuclear_repulsion(charges, coords):
    # nuclear repulsion energy in Hartree
    Vnn = 0.0
    for a, A in enumerate(charges):
        for b, B in enumerate(charges):
            if b > a:
                R = math.sqrt(
                    (coords[a][0] - coords[b][0])**2 +
                    (coords[a][1] - coords[b][1])**2 +
                    (coords[a][2] - coords[b][2])**2
                )
                Vnn += A * B / R
    return Vnn

# calculate the electronic energy, an expectation value 
Eel = 0.0
for m in range(nao):
    for n in range(nao):
        Eel += 0.5 * D[n,m] * (H[m,n] + F[m,n])
    
# calculate the total energy
E = Eel + nuclear_repulsion(charges, coords)
```

## Complete SCF procedure

In order to start a Hartree-Fock procedure, we just have to put what we have done in a loop to run over a predefined number of iterations.

```python

import numpy as np
from scipy import linalg

# Constants and parameters
nelec = 10 # number of electrons
nao = S.shape[0] # number of basis functions
scf_max_iter = 30 # maximum number of SCF iterations
conv_tol = 1e-8 # convergence threshold

# Read integrals
S = np.load('S.h2o.sto3g.npy') # overlap integrals
T = np.load('T.h2o.sto3g.npy') # kinetic energy integrals
V = np.load('V.h2o.sto3g.npy') # nuclear attraction integrals
H = np.load('H.h2o.sto3g.npy') # (core) one-electron Hamiltonian
G = np.load('G.h2o.sto3g.npy') # two-electron integrals

# generate transformation matrix X
X = linalg.sqrtm(linalg.inv(S)) 

# The density matrix with an initial guess
D = make_guess(nao)

for iter in range(scf_max_iter+1):
    	
    # Calculate the Fock matrix
    F = buildF(H, G, D)

    # Solve the Roothaan-Hall equation
    Fp = X @ F @ X
    e, Cp = np.linalg.eigh(Fp)
    C = X @ Cp  

    # calculate the electronic energy, an expectation value 
    Eel = 0.0
    for m in range(nao):
        for n in range(nao):
            Eel += 0.5 * D[n,m] * (H[m,n] + F[m,n])

    # calculate the total energy
    E = Eel + nuclear_repulsion(charges, coords)

    # save the old density matrix for comparison
    D_old = D.copy()

    # form a new density matrix D from the molecular orbitals C 
    for m in range(nao):
        for n in range(nao):
            D[m,n] = 0.0
            for a in range(int(nelec/2)):
                D[m,n] += 2 * (C[m,a] * C[n,a])

    # Calculate convergence
    conv = np.max(np.abs(D - D_old))

    # skip 1th iter
    if iter > 0:
        print(f'Iteration {iter:3}: energy = {E:12.10f}, convergence = {conv:10.4e}')

    if iter > 0 and conv <= conv_tol:
        print(f"The calculation is converged after {iter:3} iterations.")
        break 
    if iter == scf_max_iter and conv > conv_tol:
        print("The calculation is not converged!")  

```

We obtain the energy at each step in SCF procedure as follows

```
Iteration   1: energy = -73.2285323930, convergence = 1.7533e+00
Iteration   2: energy = -74.9466685767, convergence = 1.3779e-01
Iteration   3: energy = -74.9609794584, convergence = 4.2648e-02
Iteration   4: energy = -74.9616482154, convergence = 1.4074e-02
Iteration   5: energy = -74.9617359818, convergence = 5.2466e-03
Iteration   6: energy = -74.9617508689, convergence = 2.0511e-03
Iteration   7: energy = -74.9617535556, convergence = 8.3477e-04
Iteration   8: energy = -74.9617540501, convergence = 3.4784e-04
Iteration   9: energy = -74.9617541417, convergence = 1.4972e-04
Iteration  10: energy = -74.9617541587, convergence = 6.4489e-05
Iteration  11: energy = -74.9617541619, convergence = 2.7785e-05
Iteration  12: energy = -74.9617541625, convergence = 1.1973e-05
Iteration  13: energy = -74.9617541626, convergence = 5.1596e-06
Iteration  14: energy = -74.9617541626, convergence = 2.2236e-06
Iteration  15: energy = -74.9617541626, convergence = 9.5832e-07
Iteration  16: energy = -74.9617541626, convergence = 4.1302e-07
Iteration  17: energy = -74.9617541626, convergence = 1.7800e-07
Iteration  18: energy = -74.9617541626, convergence = 7.6717e-08
Iteration  19: energy = -74.9617541626, convergence = 3.3064e-08
Iteration  20: energy = -74.9617541626, convergence = 1.4250e-08
Iteration  21: energy = -74.9617541626, convergence = 6.1417e-09
The calculation is converged after  21 iterations.
```

Although our initial guess is not great, the SCF procedure gradually improves the density and the wavefunction. The first iteration yields an energy of approximately -73.22853 Hartree, and the second iteraction lowered this significantly to -74.94667 Hartree, gaining roughly 1.72 Hartree (or 1079 kcal/mol, a huge improvement!) of additional stabilization. By the 21st iteration, the total energy converges to -74.9617541626 Hartree, with differences in the density matrix on the order of `1e-9`. While the total energy appears to converge by the 14th iteration, the density matrix had not fully converged yet, highlighting that the change in the density matrix is a more sensitive and reliable indicator for SCF convergence. 

Unpack the eigenvalues of the final Fock matrix, we obtain the energy of each molecular orbitals in Hartree, i.e., the energy of one electron in the orbital. 

```
MO #1: energy = -20.235119550139746.
MO #2: energy = -1.26077940974714.
MO #3: energy = -0.623082826896677.
MO #4: energy = -0.4411608808412172.
MO #5: energy = -0.38717843286589276.
MO #6: energy = 0.592813447584928.
MO #7: energy = 0.7522187844865371.
```

Notice that the sum of all of the occupied MO energies yields a higher electronic energy because electron-electron repulsion is double counted. Additionally, the five occupied MOs all have negative energies. So, their electrons are bound within the molecule. The unoccupied MOs (called “virtual” MOs) all have positive energies, meaning that the molecule will not spontaneously accept an electron from another source.

## References
1. Lecture notes from *Hartree Fock theory* class from Professor Devin Matthews (SMU).
2. [Hartree-Fock Calculation for Water](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/The_Live_Textbook_of_Physical_Chemistry_(Peverati)/28%3A_The_Chemical_Bond_in_Polyatomic_Molecules/28.02%3A_Hartree-Fock_Calculation_for_Water)
3. Computational Chemistry: Introduction to the Theory and Applications of Molecular and Quantum Mechanics, 2nd Edition, Errol G. Lewars (2016).
