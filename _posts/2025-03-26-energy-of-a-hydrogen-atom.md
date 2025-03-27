---
title: Writing a New Post
categories: [Blogging, Tutorial]
tags: [quantum-chemistry]
math: true
---

The most exciting part in quantum chemistry is solving the wavefunction of a hydrogen atom. Because it is the only case that we can get an analytical solution in terms of energy and atomic functions. For all the other atoms, we have to cheat or make some approximations. The hydrogen’s wavefunction also address the atomic orbitals and put the fundamental solution for many approximations in quantum chemistry. 

Let us talk about the hydrogen atom. Hydrogen is the first element in the periodic table. This means it has only one electron interacting with a proton. In quantum mechanics, the energy and quantum state of a system are solutions of the Schrödinger equation. The Schrödinger equation in atomic unit for the hydrogen atom which consists of one electron and one nucleus is as follows

$$
-\frac{1}{2}\nabla^2\psi-\frac{1}{r}\psi=E\psi
$$

where $\nabla^2$ is the Laplace operator, which is defined in Cartesian coordinate as

$$
\nabla^2=\frac{\partial}{\partial x}+\frac{\partial}{\partial y}+\frac{\partial}{\partial z}
$$

It is convenient to acknowledge the spherical symmetry of the problem by expressing the derivatives in terms of spherical polar coordinates. Standard manipulation of the differentials leads to the following expression for the laplacian operator

$$
\nabla^2=\frac{1}{r^2}\frac{\partial}{\partial r}\bigg(r^2\frac{\partial}{\partial r}\bigg)+\frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\bigg(\sin\theta\frac{\partial}{\partial\theta}\bigg)+\frac{1}{r^2\sin^2\theta}\frac{\partial^2}{\partial \phi^2}
$$

This expression looks scary, where it involves three variables $r, \phi$, and $\theta$ represented in heterogenous patterns. However, we can see that the radial component $r$ does not mix with the angular components ($\phi,\theta$). We can anticipate that the Schrödinger equation for the hydrogen atom is separable into angular and radial parts. Therefore, we attempt a solution of the form

$$
\psi(r,\theta,\phi)=R(r)Y(\theta,\phi)
$$

Putting this into Equation …, we obtain

$$
-\frac{1}{2}\bigg[\frac{Y}{r^2}\frac{\partial}{\partial r}\bigg(r^2\frac{\partial R}{\partial r}\bigg)+\frac{R}{r^2\sin\theta}\frac{\partial}{\partial\theta}\bigg(\sin\theta\frac{\partial Y}{\partial\theta}\bigg)+\frac{R}{r^2\sin^2\theta}\frac{\partial^2Y}{\partial\phi^2}\bigg]-\frac{1}{r}RY=ERY
$$

Diving by $YR$ and multiplying by $-2r^2$, we get

$$
\bigg[\frac{1}{R}\frac{\partial}{\partial r}\bigg(r^2\frac{\partial R}{\partial r}\bigg)+2r+2Er^2\bigg]+\frac{1}{Y}\bigg[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\bigg(\sin\theta\frac{\partial Y}{\partial\theta}\bigg)+\frac{1}{\sin^2\theta}\frac{\partial^2Y}{\partial\phi^2}\bigg]=0
$$

Obviously, the term in the first square bracket depends only on $r$, whereas the remainder depends only on $\theta$ and $\phi$; accordingly, each must be a constant. Let say, they equal to $l(l+1)$, which we will see later that $l$ corresponds to the angular momentum quantum number.

$$
\frac{1}{R}\frac{\partial}{\partial r}\bigg(r^2\frac{\partial R}{\partial r}\bigg)+2r+2Er^2=l(l+1)\\
\frac{1}{Y}\bigg[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\bigg(\sin\theta\frac{\partial Y}{\partial\theta}\bigg)+\frac{1}{\sin^2\theta}\frac{\partial^2Y}{\partial\phi^2}\bigg]=l(l+1)
$$

In this article, we will focus on the first equation, so-called radial equation. At this stage, we write $u=rR$, so that $R=u/r$, $\partial R/\partial r=[r(\partial u/\partial r)-r]/r^2$, and $(\partial/\partial r)[r^2(\partial R/\partial r)]=r\partial^2u/\partial r^2$, and hence

$$
-\frac{1}{2}\frac{\partial^2u}{\partial r^2}+\bigg[-\frac{1}{r}+\frac{l(l+1)}{2r^2}\bigg]u=Eu
$$

The term inside the square bracket is called effective potential energy

$$
\tilde V=-\frac{1}{r}+\frac{l(l+1)}{2r^2}
$$

and it has a physical meaning. The first part is the attractive Coulomb potential energy created by … The second part is a repulsive contribution that corresponds to the existence of a centrifugal force that impels the electron away from the nucleus by virtue of its motion. The electron tends to fling it away from the vicinity of the nucleus, and there is a competition between this effect and the attract part of the potential. At very short distances from the nucleus, the repulsive component tends more strongly to infinity (as $1/r^2$) than the attractive part (which varies as $1/r$), and the former dominates. Plot the effective potential with respect to $r$ with different values of $l$.

Come back to our task to solve the equation … As you can notice, this is a second order differential equation for the function $u(r)$. Complete solutions of the radial wave equation can be found in a variety of ways, The book ABC has a good instrution using abc method to solve equation …. To solve differential equations in Python, the scipy.integrate module is commonly used, particularly the odeint function. The idea is to use odeint() to evaluate the first and second derivatives of the wavefunction $u$, which are subsequently used in the numerical integration. You can find the implementation somewhere in the internet (perhaps ChatGPT?). However, I find it less straightforward, instead, we will evaluate the second derivative of $u$ by using finite difference. The Python implementation is adapted from this article. 

The finite difference method is an approach to solve differential equations numerically. The crux of the scheme lies in approximating the differential operator by the definition of a derivative

$$
\frac{du(r)}{dr}\approx\frac{u(r+h)-u(r)}{h}
$$

In the finite difference scheme, the domain of the function is discretized with some finite step $h$. As a result, the convergence is quite sensitive to the magnitude of $h$. By the same manner, the second derivative is defined as

$$
\frac{d^2u(r)}{dr^2}\approx\frac{\frac{u(r+h)-u(r)}{h}-\frac{u(r)-u(r-h)}{h}}{h}=\frac{u(r+h)-2u(r)+u(r-h)}{h^2}
$$

For the second derivative, we need to take into account at least three points of the discretized domain, which we chose $r-h,r,$ and $r+h$. Consider our problem

$$
-\frac{1}{2}\frac{\partial^2u}{\partial r^2}+\tilde Vu=Eu
$$

We discretize the domain using a regular step of size $h$. The desired solution is the list of values $\psi(0),\psi(h),\psi(2h),\cdots,\psi((N+1)h)$, which we can index using an integer $i$. For each of these values, we can write down an equation of the form

$$
-\frac{1}{2}\frac{\partial^2u_1}{\partial r^2}+\tilde Vu_1=Eu_1
$$

where $N+2$ is the total number of points in the discretized domain. Using the approximation of the second derivative shown above, we have that

$$
\frac{-u_{i+1}+2u_i-u_{i-1}}{2h^2}+\tilde Vu_i=Eu_i\quad\forall i\in\{1,\cdots,N\}
$$

This system of equations can be regarded as a single matrix equation

$$
\left[\begin{array}{cc}
\frac{1}{h^2}+\tilde V & \frac{-1}{2h^2} \\
\frac{-1}{2h^2} & \frac{1}{h^2}+\tilde V &\frac{-1}{2h^2}\\
&\frac{-1}{2h^2}&\frac{1}{h^2}+\tilde V &\ddots\\
&&\ddots&\ddots&\frac{-1}{2h^2}
\\
&&&\frac{-1}{2h^2}&\frac{1}{h^2}+\tilde V 
\end{array}\right]\left[\begin{array}{cc}
u_1\\u_2\\u_3\\\vdots\\u_N
\end{array}\right]=E\left[\begin{array}{cc}
u_1\\u_2\\u_3\\\vdots\\u_N
\end{array}\right]
$$

which is in the form of an eigenvalue problem. Solving this, we find a set of eigenvectors $u$ representing the solution, and a corresponding set of eigenvalues $E$ which represent the energy levels. 

It is pivotal to recognize the boundary conditions. The wavefunction must satisfy the Dirichlet boundary conditions $u_0=u_{N+1}=0$. This would imply two separate but trivial equations for the values $u_0$ and $u_{N+1}$. Moreover, as these values are zero by design, the differential operator is not affected by their presence. Hence, zero boundary conditions can be implemented by simply disregarding the equations for $u_0$ and $u_{N+1}$ completely, and only retaining the $N$ equations inside the discretized domain. 

Let us start implementing the finite difference method to find the eigenvectors and eigenvalues of the Schrodinger equation for the hydrogen atom. The first matrix in equation … consist of the discretized Laplace operator and the effective potential terms in a tridiagonal matrix. This matrix represents the Hamiltonian matrix to be diagonalized in order to solve the eigenproblem. Its eigenvalues are directly the energy states $E$, and the corresponding eigenvectors $u$ are related to radial part of the hydrogen wavefunction $R$ through the simple substitution we adopted earlier, $u=rR$. Remember the matrix takes in the variables:  $r$ is the radial coordinate, and $l$ is the angular momentum number.

```python
import numpy as np

def build_hamiltonian(r, l):
    h = r[1] - r[0]
    N = len(r)
    V_tilde = -1/r + l*(l+1)/(2*r**2)
    V_tilde = np.diag(V_tilde)
    main_diag = 1.0/h**2 * np.identity(N) + V_tilde
    off_diag = -1.0/(2*h**2) * (np.eye(N, k=1) + np.eye(N, k=-1))
    hamiltonian = main_diag + off_diag
    return hamiltonian

# Construct Hamiltonian matrix
N = 2000
l = 0
r = np.linspace(100, 0.0, N, endpoint=False)
hamiltonian = build_hamiltonian(r, l)
```

Now that we have the Hamiltonian matrix, we can proceed to solve the eigenproblem. Since the number of eigenvalues obtained through the diagonalization would be equal to the number of points $N$, there will be a number of positive eigenvalues. We are interested in the bound states with negative energies, and hence we limit ourselves to a couple dozen eigenstates with the smallest magnitude, in order to avoid the high energy solutions. Once computed, the eigenvalue and eigenvector pairs are to be sorted in ascending order.

```python
# solve eigenproblem
eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)

# sort eigenvalues and eigenvectors
eigenvectors = np.array([x for _, x in sorted(zip(eigenvalues, eigenvectors.T), key=lambda pair: pair[0])])
eigenvalues = np.sort(eigenvalues)
```

Since we chose $l=0$, we are sampling the hydrogen $s$ states ($1s,2s,3s,\cdots$). The exact energy for a hydrogen atom in eV is well-known as

$$
E_n=\frac{-13.6}{n^2}\quad n=1,2,\cdots
$$

```markdown
![Desktop View](https://github.com/laiducanh/laiducanh.github.io/tree/master/assets/img/posts/hydrogen_energy.png)
```

that agrees with our numerical solutions. You can tweak the value of $l$, but the same energies are obtained whatever the value of $l$, except for the removing some energy states. That’s because for any principal quantum number $n$, the permitted values of $l$ are $0,1,\cdots,n-1$. Hence, if for instance we choose $l=1$, the lowest energy solution is the one with $n=2$. 

In principal, the energy in hydrogen atoms depends only on the principal quantum number $n$. Because $n$ are integers, the energy is also quantized. In addition, the higher values of $n$, the smaller gaps between two successive energy levels. You may also notice that not all the energy levels are found, especially for higher states. This is caused by the inadequacy of a regularly spaced grid in sampling the $1/r$ potential. Ideally either a coordinate transformation, or an adaptive grid would need to be used for a better sampling.