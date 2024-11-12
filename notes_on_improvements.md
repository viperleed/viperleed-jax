# Notes on possible improvements/optimizations

## Considering symmetry earlier – DONE
We now consider the symmetry (obatined from the viperleed slab) to reduce the
number of parameters needed in the optimization. The optimizer is given a vector
of irreducible parameters to optimize (e.g. only one x,y,z for a set of multiple
symmetry-linked atoms). This vector is then expanded to the full set parameters
(e.g. x,y,z for all atoms) and used as the input for the delta calculation.
However, since we know that some displacements can only occur in a concerted
fashion, we could potentially simplify the calculation during setup. One could,
for example, use a linear combination of the tensors for changes that can only
occur together. This may save significant amounts of computation and memory.

## Full V0r support – DONE
Due to the implementation, the parameter v0r can currently only take values that
are a multiple of the used energy step. The R-factor depends strongly on V0r, so
this is a considerable restriction (that also exists in TensErLEED!).
If we implement the interpolation algorithm differently, we could shift the knot
points at every evaluation, allowing floating point V0r values.

## Have another look at Pauls Gaunt coeffs
Paul mentioned that his new implementation is a bit slower than the old one.
Give this another look.

## Parallel I/O for reading Tensors
Reading the tensor files takes a long time. We should see if we can parallelize
this or speed it up otherwise.

## Spherical harmonic - DONE
Using symmetries of the Gaunt coefficients and spherical harmonics, it should be
possible to calculate only half as many qunatum number combinations as we
currently do.
The CSUM coeffs are symmetric (execpt for prefactors), and spherical harmonics
follow Y_(l,-m) = (-1)^m Y_(l,m)^*.

*Note:* we tried to look into this. Turns out the spherical harmonics are
calculated in a very efficient manner in JAX and there is little to no
overhead for calculating more combinations, as all possible l,m combinations
up to the max used l,m are calculated and cached anyhow in an iterative scheme.
