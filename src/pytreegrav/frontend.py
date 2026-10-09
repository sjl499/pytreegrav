import numpy as np
import warnings
from numpy import zeros_like, zeros
from .kernel import *
from .octree import *
from .dynamic_tree import *
from .treewalk import *
from .grouped_treewalk import (
    AccelTarget_grouped,
    FieldsTarget_grouped,
    PotentialTarget_grouped,
    TidalTensorTarget_grouped,
    _morton_order,
    _get_core,
)
from .bruteforce import *
from .bruteforce import _PotentialTarget_options_serial, _PotentialTarget_options_parallel
from .bruteforce_symmetric import Potential_bruteforce_symmetric, Accel_bruteforce_symmetric, SYMMETRIC_NMIN
from .misc import *


def _f64(a):
    """Cast to float64 without collapsing a length-1 array to a scalar.

    ``np.float64(a)`` looks like a dtype cast, but on any array of size 1 numpy applies its *scalar constructor* instead and returns a 0-d float64 (deprecated since numpy 1.25, fixed only in 2.4). A caller passing a single particle therefore had ``m`` and ``softening`` silently turned into scalars, and the jitclass then failed to type ``masses[oi]`` (github issue #31). ``asarray`` never does this.
    """
    return np.asarray(a, dtype=np.float64)


def checkTreeQuadrupoles(tree, quadrupole):
    """Raise ValueError if a quadrupole treewalk is requested on a tree that lacks them.

    The walk kernels index tree.Quadrupoles based only on the walk's quadrupole flag, but that array is allocated only when the tree itself was built with quadrupole=True. The mismatch is an out-of-bounds read, which numba does not bounds-check, so without this guard it segfaults instead of raising.
    """
    if quadrupole and not tree.HasQuads:
        raise ValueError(
            "quadrupole=True requires a tree built with quadrupole moments: pass a tree from "
            "ConstructTree(..., quadrupole=True), or evaluate with quadrupole=False."
        )


def valueTestMethod(method):
    """Raise TypeError/ValueError unless method is one of 'adaptive', 'bruteforce', 'tree'."""
    methods = ["adaptive", "bruteforce", "tree"]

    ## check if method is a str
    if type(method) != str:
        raise TypeError("Invalid method type %s, must be str" % type(method))

    ## check if method is a valid method
    if method not in methods:
        raise ValueError("Invalid method %s. Must be one of: %s" % (method, str(methods)))


def _potential_kernel_id(softening_kernel):
    kernels = ("cubic_spline", "wendland_c2")
    if not isinstance(softening_kernel, str) or softening_kernel not in kernels:
        raise ValueError("softening_kernel must be 'cubic_spline' or 'wendland_c2'")
    return kernels.index(softening_kernel)


def _potential_targets(pos, softening):
    pos = np.ascontiguousarray(np.atleast_2d(_f64(pos)))
    if pos.ndim != 2 or pos.shape[1] != 3 or not np.all(np.isfinite(pos)):
        raise ValueError("positions must be finite with shape (N, 3)")
    softening = zeros(len(pos)) if softening is None else np.atleast_1d(_f64(softening))
    if softening.shape != (len(pos),) or not np.all(np.isfinite(softening)) or np.any(softening < 0):
        raise ValueError("softening must be finite, nonnegative, with shape (N,)")
    return pos, np.ascontiguousarray(softening)


def _potential_masses(m, n):
    m = np.atleast_1d(_f64(m))
    if m.shape != (n,) or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("source masses must be finite, nonnegative, with shape (N,)")
    return np.ascontiguousarray(m)


def _potential_controls(G, theta, group_size):
    if not np.isscalar(G) or not np.isfinite(G) or G <= 0:
        raise ValueError("selected potential options require finite G > 0")
    if not np.isscalar(theta) or not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    if (
        isinstance(group_size, (bool, np.bool_))
        or not isinstance(group_size, (int, np.integer))
        or group_size < 1
    ):
        raise ValueError("group_size must be a positive integer")


def _potential_indices(self_index, n_target, n_source, allow_self=False):
    if self_index is None:
        return np.full(n_target, -1, dtype=np.int64)
    if isinstance(self_index, str):
        if self_index != "self" or not allow_self or n_target != n_source:
            raise ValueError("self_index='self' requires self-evaluation with known source order")
        return np.arange(n_target, dtype=np.int64)
    indices = np.asarray(self_index)
    if indices.shape != (n_target,) or indices.dtype.kind not in "iu":
        raise ValueError("self_index must be an integer array with shape (N_target,)")
    if (indices.dtype.kind == "i" and np.any(indices < -1)) or np.any(indices >= n_source):
        raise ValueError("self_index entries must be -1 or valid original source indices")
    return np.array(indices, dtype=np.int64, copy=True)


def _check_potential_tree(tree):
    if not isinstance(tree, Octree) or not tree.RadixBuilt or not tree.HasMoments:
        raise ValueError("selected potential options require a static radix/Morton tree with moments")
    if tree.HasUnresolvedPoints:
        raise ValueError("selected potential options cannot use a tree with unresolved distinct radix points")
    n = tree.NumParticles
    if n == 0:
        raise ValueError("selected tree evaluation requires at least one source")
    _potential_targets(tree.Coordinates[:n], tree.Softenings[:n])
    _potential_masses(tree.Masses[:n], n)


def _check_potential_mapping(pos, indices, source_pos, inverse=None):
    matched = indices >= 0
    source_index = indices[matched]
    if inverse is not None:
        source_index = inverse[source_index]
    # Validate an explicit claim of identity; never discover identity by position.
    # Requiring the same position also guarantees the containing node is opened.
    if not np.array_equal(pos[matched], source_pos[source_index]):
        raise ValueError("mapped targets must equal their source positions; use -1 for independent targets")


def _finite_potential(phi):
    if not np.all(np.isfinite(phi)):
        raise ValueError(
            "nonfinite potential: retained zero-separation/zero-softening pair "
            "or arithmetic outside the float64 range"
        )
    return phi


def _potential_tree_values(
    pos, soft, tree, indices, kernel_id, identity, G, theta, group_size, parallel, quadrupole, inverse=None
):
    checkTreeQuadrupoles(tree, quadrupole)
    if identity:
        if inverse is None:
            inverse = np.empty(tree.NumParticles, dtype=np.int64)
            inverse[tree.TreewalkIndices] = np.arange(tree.NumParticles, dtype=np.int64)
        _check_potential_mapping(pos, indices, tree.Coordinates, inverse)
    core = _get_core(True, False, False, quadrupole, parallel, kernel_id, identity)
    return _finite_potential(core(pos, soft, tree, group_size, theta, G, 1, indices)[:, 0])


def _potential_options(
    pos_target,
    pos_source,
    m_source,
    softening_target,
    softening_source,
    G,
    theta,
    tree,
    return_tree,
    parallel,
    method,
    quadrupole,
    group_size,
    softening_kernel,
    self_index,
    self_evaluation=False,
):
    """Validated opt-in dispatch; the existing default paths remain unchanged."""
    valueTestMethod(method)
    kernel_id = _potential_kernel_id(softening_kernel)
    _potential_controls(G, theta, group_size)
    pos_target, softening_target = _potential_targets(pos_target, softening_target)
    supplied_tree = tree is not None
    if supplied_tree:
        _check_potential_tree(tree)
        if method == "bruteforce":
            raise ValueError("a supplied tree requires method='tree' or 'adaptive'")
        n_source = tree.NumParticles
    else:
        if pos_source is None or m_source is None:
            raise ValueError("pass source positions and masses, or a source tree")
        pos_source, softening_source = _potential_targets(pos_source, softening_source)
        n_source = len(pos_source)
        m_source = _potential_masses(m_source, n_source)
    identity = self_index is not None
    indices = _potential_indices(
        self_index, len(pos_target), n_source, self_evaluation and not supplied_tree,
    )
    if self_evaluation and not supplied_tree and not identity:
        indices = np.arange(n_source, dtype=np.int64)  # legacy self-direct still excludes i == j
    if identity and not supplied_tree:
        _check_potential_mapping(pos_target, indices, pos_source)
    if method == "adaptive":
        if supplied_tree:
            method = "tree"
        elif self_evaluation:
            method = "tree" if len(pos_target) > (4000 if parallel else 1000) else "bruteforce"
        else:
            method = "tree" if len(pos_target) * n_source > 10**6 else "bruteforce"
    if method == "bruteforce":
        direct = _PotentialTarget_options_parallel if parallel else _PotentialTarget_options_serial
        phi = _finite_potential(
            direct(
                pos_target, softening_target, pos_source, m_source, softening_source,
                indices, kernel_id, identity, self_evaluation, G,
            )
        )
    else:
        if tree is None:
            if n_source == 0:
                raise ValueError("selected tree evaluation requires at least one source")
            tree = ConstructTree(pos_source, m_source, softening_source, quadrupole=quadrupole)
            _check_potential_tree(tree)
        order = _morton_order(pos_target) if len(pos_target) else np.empty(0, dtype=np.int64)
        values = _potential_tree_values(
            pos_target[order], softening_target[order], tree, indices[order], kernel_id, identity,
            G, theta, group_size, parallel, quadrupole,
        )
        phi = np.empty_like(values)
        phi[order] = values
    return (phi, tree) if return_tree else phi


def warn_if_coincident_positions(tree, softening=None):
    """Warn if the build found particle positions coincident to floating-point precision.

    Asks the tree rather than scanning the input: the build cannot subdivide coincident points forever, so it already detects them exactly, and reading its flag is free. The np.unique-per-dimension scan this replaces cost over half the build time at N=1e7 and was wrong besides, testing whether any single *coordinate* repeated rather than any *position* -- so it warned about any lattice or grid whose positions were in fact all distinct.
    """
    if not tree.HasCoincidentPoints:
        return

    if softening is not None and np.any(softening > 0):
        warnings.warn(
            "Warning: Particle positions are non-unique. Softening will determine the answer for overlapping particles."
        )
        return

    warnings.warn(
        "Warning: Particle positions are non-unique. The answer will be singular or garbage for overlapping particles."
    )
    return


def ConstructTree(
    pos,
    m=None,
    softening=None,
    quadrupole=False,
    vel=None,
    compute_moments=True,
    morton_order=True,
    radix=True,
):
    """Builds a tree containing particle data, for subsequent potential/field evaluation

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like or None, optional
        shape (N,) array of particle masses - if None then zeros will be used (e.g. if all you need the tree for is spatial algorithms)
    softening: array_like or None, optional
        shape (N,) array of particle softening lengths - these give the radius of compact support of the M4 cubic spline mass distribution of each particle
    quadrupole: bool, optional
        Whether to store quadrupole moments (default False)
    vel: array_like or None, optional
        shape (N,3) array of particle velocities. If provided, a DynamicOctree that also stores node-averaged velocities is built (used by the velocity correlation/structure functions); if None, a static Octree is built (default None)
    compute_moments: bool, optional
        Whether to compute node multipole moments (centers of mass, masses, softenings, and quadrupoles if enabled). Set False to build only the spatial structure, e.g. for purely geometric queries (default True; forced False when m is None)
    morton_order: bool, optional
        Whether to store particles in Morton (depth-first traversal) order for cache-efficient treewalks (default True)
    radix: bool, optional
        Whether to use the radix-sort tree build (faster, default True). Set False to use the legacy insertion build. Ignored when vel is provided (dynamic tree always uses insertion), or when morton_order is False.

    Returns
    -------
    tree: Octree or DynamicOctree
        tree instance built from the particle data
    """

    # Coerce here rather than trusting the caller: the jitclass reports a shape/dtype mismatch as a
    # numba TypingError deep in the build, which is unreadable. atleast_1d/2d in particular rescue a
    # single particle, whose arrays a caller can easily have had collapsed to scalars -- see _f64.
    pos = np.atleast_2d(_f64(pos))
    if m is None:
        m = zeros(len(pos))
        compute_moments = False
    m = np.atleast_1d(_f64(m))
    if softening is None:
        softening = zeros_like(m)
    softening = np.atleast_1d(_f64(softening))
    if vel is not None:
        vel = np.atleast_2d(_f64(vel))
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(m)) and np.all(np.isfinite(softening))):
        print("Invalid input detected - aborting treebuild to avoid going into an infinite loop!")
        raise

    if vel is None:
        tree = Octree(
            pos,
            m,
            softening,
            quadrupole=quadrupole,
            compute_moments=compute_moments,
            morton_order=morton_order,
            radix=radix,
        )
    else:
        tree = DynamicOctree(pos, m, softening, vel, quadrupole=quadrupole)

    # the build detects coincident positions as a side effect, so this costs nothing
    warn_if_coincident_positions(tree, softening)
    return tree


def Potential(
    pos,
    m,
    softening=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
    device="cpu",
    *,
    softening_kernel="cubic_spline",
    self_index=None,
):
    """Gravitational potential calculation

    Returns the gravitational potential for a set of particles with positions x and masses m, at the positions of those particles, using either brute force or tree-based methods depending on the number of particles.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    G: float, optional
        gravitational constant (default 1.0)
    softening: None or array_like, optional
        shape (N,) array of compact-support radii for the selected gravitational kernel (default 0); each pair uses the greater source/target radius
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, giving ~0.2% RMS acceleration error on a Plummer sphere; 0.5 gives ~0.1%)
    parallel: bool, optional
        If True, will parallelize the force summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8, ~2-3x faster than 1 at equal-or-better accuracy; much larger values slow down again as group bounding boxes open more nodes). 1 reproduces the per-particle walk. Only affects the tree method.

    device: str, optional
        'cpu' (default) or 'cuda'. 'cuda' needs pytreegrav[cuda] and an NVIDIA GPU, and covers the monopole tree and brute-force methods. It is float32, but its error against the CPU path stays below theta's own truncation error. Uploads the tree (or sources) on every call, which for gravity costs more than the walk does -- measured ~4x faster than 32 CPU threads at N=2.2e7, against ~32x with the tree already resident -- so for repeated evaluation hold a pytreegrav.cuda.CudaPotential/CudaAccel or their Bruteforce counterparts instead.

    softening_kernel: str, optional
        "cubic_spline" (default) or "wendland_c2". Wendland support H is three
        times the Plummer-equivalent softening, with central potential -3*G*m/H.
        Selecting Wendland alone preserves each path's legacy coincidence policy.
    self_index: None, "self", or integer array, optional
        None preserves legacy exclusion. "self" excludes each input particle's
        own source index and requires tree=None. A shape (N,) integer array
        excludes the named original source index per target; -1 excludes none.
        Mapped targets must have exactly the stored source's position. Distinct
        coincident particles are retained; an unsoftened retained pair raises.

    Notes
    -----
    New kernel/identity options use CPU float64 potentials only, with finite
    nonnegative masses/supports and finite G > 0. Tree evaluation requires a
    static radix/Morton tree with moments and resolved geometry. A supplied
    tree is authoritative: mapping indices refer to its original input order,
    adaptive uses that tree, and explicit brute force is rejected. group_size=1
    gives the ungrouped optional walk. All optional direct calls use a general
    target sum rather than the symmetric optimization.

    G and input units must be consistent. The result is specific potential,
    without a factor one half; full self energy is 0.5*sum(m*phi). Validate theta
    against potential/energy errors for the application, not force-error figures.

    Returns
    -------
    phi: array_like
        shape (N,) array of potentials at the particle positions
    """

    if (
        not isinstance(softening_kernel, str)
        or softening_kernel != "cubic_spline"
        or self_index is not None
    ):
        if device != "cpu":
            raise ValueError("selected kernel/identity options support device='cpu' only")
        return _potential_options(
            pos, pos, m, softening, softening, G, theta, tree, return_tree,
            parallel, method, quadrupole, group_size, softening_kernel, self_index, True,
        )

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    # Coerce once, up front, so every method sees the same thing. Only the tree path used to coerce
    # (via ConstructTree), so lists worked with method="tree" and raised a numba TypingError with
    # method="bruteforce"; asarray is a no-op for arrays that are already float64.
    pos = np.atleast_2d(_f64(pos))
    m = np.atleast_1d(_f64(m))
    if softening is None:
        softening = np.zeros_like(m)
    softening = np.atleast_1d(_f64(softening))

    # figure out which method to use
    if method == "adaptive":
        # A supplied tree pins the method: brute force cannot use it and would silently sum only the
        # particles in pos, a different quantity whenever the tree holds a different set. Otherwise,
        # threaded brute force stays competitive to larger N -- 4000 is where the parallel curves cross
        # on 32 threads (Plummer, theta=0.7): at N=3447 brute-force accel is 0.57 us/particle against
        # the tree's 0.67, by N=6092 it is 0.95 against 0.68. Potential crosses nearer 5500, so this
        # favours acceleration, the more common call and the steeper curve to be wrong about.
        if tree is not None or len(pos) > (4000 if parallel else 1000):
            method = "tree"
        else:
            method = "bruteforce"

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and quadrupole:
        raise ValueError("device='cuda' is monopole only; pass quadrupole=False")

    if method == "bruteforce":  # we're using brute force
        if device == "cuda":
            from .cuda import CudaPotentialBruteforce  # lazy: numba-cuda is an optional extra

            phi = CudaPotentialBruteforce(pos, m, softening)(pos, softening, G=G)
        elif parallel:
            # the symmetrized kernel runs two parallel regions and per-thread buffers;
            # that only pays for itself once there are enough interactions to amortize it
            if len(pos) >= SYMMETRIC_NMIN:
                phi = Potential_bruteforce_symmetric(pos, m, softening, G=G)
            else:
                phi = Potential_bruteforce_parallel(pos, m, softening, G=G)
        else:
            phi = Potential_bruteforce(pos, m, softening, G=G)
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                _f64(pos),
                _f64(m),
                _f64(softening),
                quadrupole=quadrupole,
            )  # build the tree if needed
            idx = tree.TreewalkIndices  # built from pos, so its walk order already indexes pos
        else:
            # SORT pos; don't apply the tree's stored permutation. TreewalkIndices is a fixed sigma
            # over whatever built the tree, and is not an involution -- pos already in tree order
            # becomes X[sigma^2]: right answer, but the groups are no longer spatially compact, so
            # grouping's acceptance padding inflated ~94x on clustered data: 15 s -> 285 s.
            # A larger supplied tree raises IndexError outright. Sorting is idempotent, fixing both.
            idx = _morton_order(_f64(pos))
        checkTreeQuadrupoles(tree, quadrupole)

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        pos_sorted = np.take(pos, idx, axis=0)
        h_sorted = np.take(softening, idx)

        if device == "cuda":
            from .cuda import CudaPotential  # lazy: numba-cuda is an optional extra

            phi = CudaPotential(tree)(pos_sorted, h_sorted, G=G, theta=theta)
        else:
            # pos_sorted is in Morton order, so consecutive targets are spatially compact groups
            phi = PotentialTarget_grouped(
                pos_sorted,
                h_sorted,
                tree,
                group_size=group_size,
                G=G,
                theta=theta,
                quadrupole=quadrupole,
                parallel=parallel,
            )

        # now reorder phi back to the order of the input positions
        # Scatter back rather than np.take(phi, idx.argsort()): inverting a permutation by sorting it
        # is O(N log N) for what a scatter does in O(N), and idx is a permutation by construction.
        # Bit-identical; worth 0.65 s of a 5.5 s device='cuda' call at N=2.2e7 on a Xeon Gold 6244.
        out = np.empty_like(phi)
        out[idx] = phi
        phi = out

    if return_tree:
        return phi, tree
    else:
        return phi


def PotentialTarget(
    pos_target,
    pos_source,
    m_source,
    softening_target=None,
    softening_source=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
    *,
    softening_kernel="cubic_spline",
    self_index=None,
):
    """Gravitational potential calculation for general N+M body case

    Returns the gravitational potential for a set of M particles with positions x_source and masses m_source, at the positions of a set of N particles that need not be the same.

    Parameters
    ----------
    pos_target: array_like
        shape (N,3) array of target particle positions where you want to know the potential
    pos_source: array_like
        shape (M,3) array of source particle positions (positions of particles sourcing the gravitational field)
    m_source: array_like
        shape (M,) array of source particle masses
    softening_target: array_like or None, optional
        shape (N,) array of target compact-support radii for the selected kernel; a pair uses max(target radius, source radius)
    softening_source: array_like or None, optional
        shape (M,) array of source compact-support radii for the selected kernel
    G: float, optional
        gravitational constant (default 1.0)
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, giving ~0.2% RMS acceleration error on a Plummer sphere; 0.5 gives ~0.1%)
    parallel: bool, optional
        If True, will parallelize the force summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8, ~2-3x faster than 1 at equal-or-better accuracy; much larger values slow down again as group bounding boxes open more nodes). 1 reproduces the per-particle walk. Only affects the tree method.

    softening_kernel: str, optional
        "cubic_spline" (default) or "wendland_c2". See Potential for support
        conventions, units and the CPU float64 potential-only support boundary.
    self_index: None or integer array, optional
        None preserves legacy coordinate exclusion. A shape (N,) integer array
        enables identity exclusion: j names original source j, and -1 means no
        corresponding source. Use all -1 for independent targets, including at
        source coordinates. Repeated indices are allowed. Mapped target positions
        must exactly equal their stored sources; no identities are inferred.
        "self" is not accepted here; use an explicit mapping for subsets/reorders.

    Notes
    -----
    For reuse, pass pos_source=None and m_source=None with tree. The tree's
    stored sources are authoritative; mappings retain its original construction
    index space. With new options, a supplied tree pins adaptive to tree and
    conflicts with explicit brute force. Rebuild after source changes. Retained
    zero-separation pairs require positive pair support or raise ValueError.
    The new options require static radix/Morton trees with computed moments and
    resolved geometry; insertion/dynamic trees and non-potential fields are not
    supported. Source masses/supports must be nonnegative and finite, with G > 0.

    Returns
    -------
    phi: array_like
        shape (N,) array of potentials at the target positions
    """

    if (
        not isinstance(softening_kernel, str)
        or softening_kernel != "cubic_spline"
        or self_index is not None
    ):
        return _potential_options(
            pos_target, pos_source, m_source, softening_target, softening_source,
            G, theta, tree, return_tree, parallel, method, quadrupole, group_size,
            softening_kernel, self_index,
        )

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    ## allow user to pass in tree without passing in source pos and m
    ##  but catch if they don't pass in the tree.
    if tree is None and (pos_source is None or m_source is None):
        raise ValueError("Must pass either pos_source & m_source or source tree.")

    if softening_target is None:
        softening_target = zeros(len(pos_target))
    if softening_source is None and pos_source is not None:
        softening_source = zeros(len(pos_source))

    # figure out which method to use
    if method == "adaptive":
        if pos_source is None or len(pos_target) * len(pos_source) > 10**6:
            method = "tree"
        else:
            method = "bruteforce"

    if method == "bruteforce":  # we're using brute force
        if parallel:
            phi = PotentialTarget_bruteforce_parallel(
                pos_target,
                softening_target,
                pos_source,
                m_source,
                softening_source,
                G=G,
            )
        else:
            phi = PotentialTarget_bruteforce(
                pos_target,
                softening_target,
                pos_source,
                m_source,
                softening_source,
                G=G,
            )
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                _f64(pos_source),
                _f64(m_source),
                _f64(softening_source),
                quadrupole=quadrupole,
            )  # build the tree if needed
        checkTreeQuadrupoles(tree, quadrupole)
        # external targets are not spatially ordered; Morton-sort them so grouping is effective
        tsort = _morton_order(_f64(pos_target))
        phi_sorted = PotentialTarget_grouped(
            _f64(pos_target)[tsort],
            _f64(softening_target)[tsort],
            tree,
            group_size=group_size,
            G=G,
            theta=theta,
            quadrupole=quadrupole,
            parallel=parallel,
        )
        phi = np.empty_like(phi_sorted)
        phi[tsort] = phi_sorted  # undo the Morton permutation

    if return_tree:
        return phi, tree
    else:
        return phi


def Accel(
    pos,
    m,
    softening=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
    device="cpu",
):
    """Gravitational acceleration calculation

    Returns the gravitational acceleration for a set of particles with positions x and masses m, at the positions of those particles, using either brute force or tree-based methods depending on the number of particles.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    G: float, optional
        gravitational constant (default 1.0)
    softening: None or array_like, optional
        shape (N,) array containing kernel support radii for gravitational softening - these give the radius of compact support of the M4 cubic spline mass distribution - set to 0 by default
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, giving ~0.2% RMS acceleration error on a Plummer sphere; 0.5 gives ~0.1%)
    parallel: bool, optional
        If True, will parallelize the force summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8, ~2-3x faster than 1 at equal-or-better accuracy; much larger values slow down again as group bounding boxes open more nodes). 1 reproduces the per-particle walk. Only affects the tree method.

    device: str, optional
        'cpu' (default) or 'cuda'. 'cuda' needs pytreegrav[cuda] and an NVIDIA GPU, and covers the monopole tree and brute-force methods. It is float32, but its error against the CPU path stays below theta's own truncation error. Uploads the tree (or sources) on every call, which for gravity costs more than the walk does -- measured ~4x faster than 32 CPU threads at N=2.2e7, against ~32x with the tree already resident -- so for repeated evaluation hold a pytreegrav.cuda.CudaPotential/CudaAccel or their Bruteforce counterparts instead.

    Returns
    -------
    g: array_like
        shape (N,3) array of acceleration vectors at the particle positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    # Coerce once, up front, so every method sees the same thing. Only the tree path used to coerce
    # (via ConstructTree), so lists worked with method="tree" and raised a numba TypingError with
    # method="bruteforce"; asarray is a no-op for arrays that are already float64.
    pos = np.atleast_2d(_f64(pos))
    m = np.atleast_1d(_f64(m))
    if softening is None:
        softening = np.zeros_like(m)
    softening = np.atleast_1d(_f64(softening))

    # figure out which method to use
    if method == "adaptive":
        # see the method-selection note in Potential
        if tree is not None or len(pos) > (4000 if parallel else 1000):
            method = "tree"
        else:
            method = "bruteforce"

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and quadrupole:
        raise ValueError("device='cuda' is monopole only; pass quadrupole=False")

    if method == "bruteforce":  # we're using brute force
        if device == "cuda":
            from .cuda import CudaAccelBruteforce  # lazy: numba-cuda is an optional extra

            g = CudaAccelBruteforce(pos, m, softening)(pos, softening, G=G)
        elif parallel:
            # see the note in Potential: small problems stay on the simpler kernel
            if len(pos) >= SYMMETRIC_NMIN:
                g = Accel_bruteforce_symmetric(pos, m, softening, G=G)
            else:
                g = Accel_bruteforce_parallel(pos, m, softening, G=G)
        else:
            g = Accel_bruteforce(pos, m, softening, G=G)
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                _f64(pos),
                _f64(m),
                _f64(softening),
                quadrupole=quadrupole,
            )  # build the tree if needed
            idx = tree.TreewalkIndices  # built from pos, so its walk order already indexes pos
        else:
            # SORT pos; don't apply the tree's stored permutation. TreewalkIndices is a fixed sigma
            # over whatever built the tree, and is not an involution -- pos already in tree order
            # becomes X[sigma^2]: right answer, but the groups are no longer spatially compact, so
            # grouping's acceptance padding inflated ~94x on clustered data: 15 s -> 285 s.
            # A larger supplied tree raises IndexError outright. Sorting is idempotent, fixing both.
            idx = _morton_order(_f64(pos))
        checkTreeQuadrupoles(tree, quadrupole)

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        pos_sorted = np.take(pos, idx, axis=0)
        h_sorted = np.take(softening, idx)

        if device == "cuda":
            from .cuda import CudaAccel  # lazy: numba-cuda is an optional extra

            g = CudaAccel(tree)(pos_sorted, h_sorted, G=G, theta=theta)
        else:
            # pos_sorted is in Morton order, so consecutive targets are spatially compact groups
            g = AccelTarget_grouped(
                pos_sorted,
                h_sorted,
                tree,
                group_size=group_size,
                G=G,
                theta=theta,
                quadrupole=quadrupole,
                parallel=parallel,
            )

        # now g is in the tree-order: reorder it back to the original order
        out = np.empty_like(g)  # scatter, not argsort; see the note in Potential
        out[idx] = g
        g = out

    if return_tree:
        return g, tree
    else:
        return g


def AccelTarget(
    pos_target,
    pos_source,
    m_source,
    softening_target=None,
    softening_source=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
):
    """Gravitational acceleration calculation for general N+M body case

    Returns the gravitational acceleration for a set of M particles with positions x_source and masses m_source, at the positions of a set of N particles that need not be the same.

    Parameters
    ----------
    pos_target: array_like
        shape (N,3) array of target particle positions where you want to know the acceleration
    pos_source: array_like
        shape (M,3) array of source particle positions (positions of particles sourcing the gravitational field)
    m_source: array_like
        shape (M,) array of source particle masses
    softening_target: array_like or None, optional
        shape (N,) array of target particle softening radii - these give the radius of compact support of the M4 cubic spline mass distribution
    softening_source: array_like or None, optional
        shape (M,) array of source particle radii - these give the radius of compact support of the M4 cubic spline mass distribution
    G: float, optional
        gravitational constant (default 1.0)
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, giving ~0.2% RMS acceleration error on a Plummer sphere; 0.5 gives ~0.1%)
    parallel: bool, optional
        If True, will parallelize the force summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8, ~2-3x faster than 1 at equal-or-better accuracy; much larger values slow down again as group bounding boxes open more nodes). 1 reproduces the per-particle walk. Only affects the tree method.

    Returns
    -------
    phi: array_like
        shape (N,3) array of accelerations at the target positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    ## allow user to pass in tree without passing in source pos and m
    ##  but catch if they don't pass in the tree.
    if tree is None and (pos_source is None or m_source is None):
        raise ValueError("Must pass either pos_source & m_source or source tree.")

    if softening_target is None:
        softening_target = zeros(len(pos_target))
    if softening_source is None and pos_source is not None:
        softening_source = zeros(len(pos_source))

    # figure out which method to use
    if method == "adaptive":
        if pos_source is None or len(pos_target) * len(pos_source) > 10**6:
            method = "tree"
        else:
            method = "bruteforce"

    if method == "bruteforce":  # we're using brute force
        if parallel:
            g = AccelTarget_bruteforce_parallel(
                pos_target,
                softening_target,
                pos_source,
                m_source,
                softening_source,
                G=G,
            )
        else:
            g = AccelTarget_bruteforce(
                pos_target,
                softening_target,
                pos_source,
                m_source,
                softening_source,
                G=G,
            )
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                _f64(pos_source),
                _f64(m_source),
                _f64(softening_source),
                quadrupole=quadrupole,
            )  # build the tree if needed
        checkTreeQuadrupoles(tree, quadrupole)
        # external targets are not spatially ordered; Morton-sort them so grouping is effective
        tsort = _morton_order(_f64(pos_target))
        g_sorted = AccelTarget_grouped(
            _f64(pos_target)[tsort],
            _f64(softening_target)[tsort],
            tree,
            group_size=group_size,
            G=G,
            theta=theta,
            quadrupole=quadrupole,
            parallel=parallel,
        )
        g = np.empty_like(g_sorted)
        g[tsort] = g_sorted  # undo the Morton permutation

    if return_tree:
        return g, tree
    else:
        return g


def TidalTensor(
    pos,
    m,
    softening=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
):
    """Tidal tensor calculation

    Returns the tidal tensor T_ij = dg_i/dx_j = -d^2 phi/(dx_i dx_j) for a set of particles with positions pos and masses m, at the positions of those particles, using either brute force or tree-based methods depending on the number of particles.

    In this convention the relative tidal acceleration of a neighbour at small separation dx is T_ij dx_j, positive eigenvalues are stretching and negative ones compressive, and the trace is -4 pi G rho: exactly zero wherever no softening kernel overlaps the evaluation point.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    softening: None or array_like, optional
        shape (N,) array containing kernel support radii for gravitational softening - these give the radius of compact support of the M4 cubic spline mass distribution - set to 0 by default
    G: float, optional
        gravitational constant (default 1.0)
    theta: float, optional
        cell opening angle used to control accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. The tidal tensor is one derivative higher than the acceleration, so its fractional error at fixed theta is correspondingly larger - use a smaller theta than you would for forces, and quadrupole=True. (default 0.7)
    parallel: bool, optional
        If True, will parallelize the summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8). 1 reproduces the per-particle walk. Only affects the tree method.

    Returns
    -------
    T: array_like
        shape (N,3,3) array of tidal tensors at the particle positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    # coerce up front so every method sees the same thing -- see the note in Accel
    pos = np.atleast_2d(_f64(pos))
    m = np.atleast_1d(_f64(m))
    if softening is None:
        softening = np.zeros_like(m)
    softening = np.atleast_1d(_f64(softening))

    # figure out which method to use
    if method == "adaptive":
        # see the method-selection note in Potential
        if tree is not None or len(pos) > (4000 if parallel else 1000):
            method = "tree"
        else:
            method = "bruteforce"

    if method == "bruteforce":  # we're using brute force
        if parallel:
            T = TidalTensor_bruteforce_parallel(pos, m, softening, G=G)
        else:
            T = TidalTensor_bruteforce(pos, m, softening, G=G)
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(pos, m, softening, quadrupole=quadrupole)  # build the tree if needed
            idx = tree.TreewalkIndices  # built from pos, so its walk order already indexes pos
        else:
            idx = _morton_order(pos)  # see the sorting note in Accel
        checkTreeQuadrupoles(tree, quadrupole)

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        T = TidalTensorTarget_grouped(
            np.take(pos, idx, axis=0),
            np.take(softening, idx),
            tree,
            group_size=group_size,
            G=G,
            theta=theta,
            quadrupole=quadrupole,
            parallel=parallel,
        )
        out = np.empty_like(T)  # back to the original order; scatter, not argsort, per Potential
        out[idx] = T
        T = out

    if return_tree:
        return T, tree
    else:
        return T


def TidalTensorTarget(
    pos_target,
    pos_source,
    m_source,
    softening_target=None,
    softening_source=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
):
    """Tidal tensor calculation for general N+M body case

    Returns the tidal tensor T_ij = dg_i/dx_j = -d^2 phi/(dx_i dx_j) sourced by a set of M particles with positions pos_source and masses m_source, at the positions of a set of N particles that need not be the same. See :func:`TidalTensor` for the sign convention.

    Parameters
    ----------
    pos_target: array_like
        shape (N,3) array of target particle positions where you want to know the tidal tensor
    pos_source: array_like
        shape (M,3) array of source particle positions (positions of particles sourcing the gravitational field)
    m_source: array_like
        shape (M,) array of source particle masses
    softening_target: array_like or None, optional
        shape (N,) array of target particle softening radii - these give the radius of compact support of the M4 cubic spline mass distribution
    softening_source: array_like or None, optional
        shape (M,) array of source particle radii - these give the radius of compact support of the M4 cubic spline mass distribution
    G: float, optional
        gravitational constant (default 1.0)
    theta: float, optional
        cell opening angle used to control accuracy; see the note in :func:`TidalTensor` (default 0.7)
    parallel: bool, optional
        If True, will parallelize the summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree of the sources (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8). 1 reproduces the per-particle walk. Only affects the tree method.

    Returns
    -------
    T: array_like
        shape (N,3,3) array of tidal tensors at the target positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    ## allow user to pass in tree without passing in source pos and m
    ##  but catch if they don't pass in the tree.
    if tree is None and (pos_source is None or m_source is None):
        raise ValueError("Must pass either pos_source & m_source or source tree.")

    pos_target = np.atleast_2d(_f64(pos_target))
    if softening_target is None:
        softening_target = zeros(len(pos_target))
    softening_target = np.atleast_1d(_f64(softening_target))
    if pos_source is not None:
        pos_source = np.atleast_2d(_f64(pos_source))
        m_source = np.atleast_1d(_f64(m_source))
        if softening_source is None:
            softening_source = np.zeros_like(m_source)
        softening_source = np.atleast_1d(_f64(softening_source))

    # figure out which method to use
    if method == "adaptive":
        if pos_source is None or len(pos_target) * len(pos_source) > 10**6:
            method = "tree"
        else:
            method = "bruteforce"

    if method == "bruteforce":  # we're using brute force
        kernel = TidalTensorTarget_bruteforce_parallel if parallel else TidalTensorTarget_bruteforce
        T = kernel(pos_target, softening_target, pos_source, m_source, softening_source, G=G)
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                pos_source, m_source, softening_source, quadrupole=quadrupole
            )  # build the tree if needed
        checkTreeQuadrupoles(tree, quadrupole)
        # external targets are not spatially ordered; Morton-sort them so grouping is effective
        tsort = _morton_order(pos_target)
        T_sorted = TidalTensorTarget_grouped(
            pos_target[tsort],
            softening_target[tsort],
            tree,
            group_size=group_size,
            G=G,
            theta=theta,
            quadrupole=quadrupole,
            parallel=parallel,
        )
        T = np.empty_like(T_sorted)
        T[tsort] = T_sorted  # undo the Morton permutation

    if return_tree:
        return T, tree
    else:
        return T


class Field:
    """A reusable gravitational field: build the tree once, evaluate many times.

    The module-level functions are self-contained by design -- ``Accel(x, m, h)`` is a complete answer with no setup -- but they pay for that on every call. Each one rebuilds the tree (or, given ``tree=``, re-derives the Morton permutation of the targets), and each returns exactly one quantity. Evaluating the potential, the field and the tidal tensor of one mass distribution therefore builds three permutations and walks the tree three times.

    ``Field`` holds the tree, the sorted source arrays and the permutation, so repeated evaluation pays for none of that again, and :meth:`evaluate` gets every requested quantity out of a *single* traversal. This is the CPU counterpart of the ``pytreegrav.cuda`` context objects, which already worked this way.

    >>> f = Field(pos, m, softening=h, theta=0.4, quadrupole=True, parallel=True)
    >>> phi = f.potential()                                  # one walk
    >>> res = f.evaluate(potential=True, accel=True)         # both from one walk
    >>> T = f.tidal(pos_target=grid)                         # at arbitrary points

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of source particle positions
    m: array_like
        shape (N,) array of source particle masses
    softening: None or array_like, optional
        shape (N,) array of kernel support radii for gravitational softening (default 0)
    G: float, optional
        gravitational constant (default 1.0)
    theta: float, optional
        opening angle; may be overridden per call (default 0.7)
    quadrupole: bool, optional
        build and use node quadrupole moments (default False). Fixed at construction, because it determines what the tree stores.
    group_size: int, optional
        targets sharing one traversal (default 8)
    parallel: bool, optional
        parallelize over groups; may be overridden per call (default False)

    softening_kernel: str, optional
        "cubic_spline" (default) or "wendland_c2", fixed at construction.
        Wendland permits potential-only evaluation. Its compact-support radius
        is H=3*epsilon_Plummer; pair supports use the maximum source/target value.

    Notes
    -----
    Source positions, masses, softenings and the selected kernel are fixed once
    built; changing them requires a new Field. Do not mutate the stored tree.
    Wendland or explicit self_index evaluations support CPU float64 potential
    only; acceleration, tidal and mixed requests raise rather than mixing kernels.
    See Potential for validation, singularity handling and legacy coincidences.
    """

    def __init__(
        self,
        pos,
        m,
        softening=None,
        G=1.0,
        theta=0.7,
        quadrupole=False,
        group_size=8,
        parallel=False,
        *,
        softening_kernel="cubic_spline",
    ):
        self.softening_kernel = softening_kernel
        self._kernel_id = _potential_kernel_id(softening_kernel)
        self._potential_tree_checked = False
        if self._kernel_id:
            pos, softening = _potential_targets(pos, softening)
            m = _potential_masses(m, len(pos))
            _potential_controls(G, theta, group_size)
            if len(pos) == 0:
                raise ValueError("Field requires at least one source")
        # coerce exactly as the functional API does -- see the note in Accel
        pos = np.atleast_2d(_f64(pos))
        m = np.atleast_1d(_f64(m))
        if softening is None:
            softening = np.zeros_like(m)
        softening = np.atleast_1d(_f64(softening))

        self.G = G
        self.theta = theta
        self.quadrupole = quadrupole
        self.group_size = group_size
        self.parallel = parallel
        self.tree = ConstructTree(pos, m, softening, quadrupole=quadrupole)

        # The source permutation is computed once here rather than per call.  TreewalkIndices is
        # built from pos, so it already indexes pos; its argsort undoes it.
        self._idx = self.tree.TreewalkIndices
        self._inv = self._idx.argsort()
        self._pos_sorted = np.take(pos, self._idx, axis=0)
        self._soft_sorted = np.take(softening, self._idx)

    def __repr__(self):
        return (
            f"<pytreegrav.Field: {self.tree.NumParticles} particles, "
            f"{'quadrupole' if self.quadrupole else 'monopole'}, theta={self.theta}, "
            f"G={self.G}, group_size={self.group_size}, parallel={self.parallel}>"
        )

    def evaluate(
        self,
        potential=False,
        accel=False,
        tidal=False,
        pos_target=None,
        softening_target=None,
        theta=None,
        parallel=None,
        group_size=None,
        *,
        self_index=None,
    ):
        """Evaluate any combination of the fields in a single tree traversal.

        Requesting several at once is substantially cheaper than one call each, because the per-interaction setup -- the distance, the softening, and with ``quadrupole=True`` the moment contraction -- is computed once and shared. Measured on 1e6 Plummer particles at 32 threads, against the equivalent separate walks: potential+accel 1.54-1.70x, accel+tidal 1.36-1.41x, all three 1.64-1.65x.

        The catch is that one traversal means one acceptance criterion, hence one ``theta`` for everything requested. The tidal tensor is a derivative higher than the potential and wants a smaller ``theta`` for equal fractional accuracy, so a fused call runs all of them at the strictest requirement; if you want the potential at 0.7 and the tidal tensor at 0.4, two calls are cheaper. When they *are* wanted at one ``theta``, fusing additionally makes them mutually consistent, since they then share an accepted-node set.

        Parameters
        ----------
        potential, accel, tidal: bool, optional
            which fields to return; at least one must be True
        pos_target: array_like or None, optional
            shape (M,3) points at which to evaluate. Defaults to the source positions.
        softening_target: array_like or None, optional
            shape (M,) floor on the softening used at each target (default 0)
        theta, parallel, group_size: optional
            per-call overrides of the values given to the constructor

        self_index: None, "self", or integer array, optional
            None retains legacy coordinate exclusion. "self" excludes the
            corresponding original source and requires pos_target=None. Otherwise
            pass shape (M,) original-source indices, using -1 for unrelated targets.
            Mapped target positions must equal their stored source positions.
            Distinct coincident sources remain and require positive pair support.
            With self_index supplied, only potential=True is supported. Self
            evaluation uses stored source softenings; an explicit softening_target
            override is rejected in the optional path.

        Returns
        -------
        dict
            the requested keys: ``"potential"`` shape (M,), ``"accel"`` shape (M,3), ``"tidal"`` shape (M,3,3), in the order of ``pos_target`` (or of the sources)
        """
        if not (potential or accel or tidal):
            raise ValueError("request at least one of potential=True, accel=True, tidal=True")
        theta = self.theta if theta is None else theta
        parallel = self.parallel if parallel is None else parallel
        group_size = self.group_size if group_size is None else group_size
        extended = self._kernel_id != 0 or self_index is not None
        if extended:
            if not potential or accel or tidal:
                raise ValueError("selected kernel/identity options support potential-only Field evaluation")
            _potential_controls(self.G, theta, group_size)
            if not self._potential_tree_checked:
                _check_potential_tree(self.tree)
                self._potential_tree_checked = True
            if pos_target is None:
                if softening_target is not None:
                    raise ValueError("self-evaluation uses the stored source softenings")
                indices = _potential_indices(self_index, self.tree.NumParticles, self.tree.NumParticles, True)
                order = self._idx
                sorted_pos, sorted_soft = self._pos_sorted, self._soft_sorted
            else:
                pos_target, softening_target = _potential_targets(pos_target, softening_target)
                indices = _potential_indices(self_index, len(pos_target), self.tree.NumParticles)
                order = _morton_order(pos_target) if len(pos_target) else np.empty(0, dtype=np.int64)
                sorted_pos, sorted_soft = pos_target[order], softening_target[order]
            values = _potential_tree_values(
                sorted_pos, sorted_soft, self.tree, indices[order], self._kernel_id, self_index is not None,
                self.G, theta, group_size, parallel, self.quadrupole, self._inv,
            )
            phi = np.empty_like(values)
            phi[order] = values
            return {"potential": phi}

        if pos_target is None:  # self-evaluation: reuse the permutation built in __init__
            sorted_pos, sorted_soft, unsort = self._pos_sorted, self._soft_sorted, self._inv
        else:
            pos_target = np.atleast_2d(_f64(pos_target))
            if softening_target is None:
                softening_target = zeros(len(pos_target))
            softening_target = np.atleast_1d(_f64(softening_target))
            # external targets are not spatially ordered; Morton-sort so grouping is effective
            tsort = _morton_order(pos_target)
            sorted_pos, sorted_soft, unsort = pos_target[tsort], softening_target[tsort], tsort.argsort()

        out = FieldsTarget_grouped(
            sorted_pos,
            sorted_soft,
            self.tree,
            potential=potential,
            accel=accel,
            tidal=tidal,
            group_size=group_size,
            G=self.G,
            theta=theta,
            quadrupole=self.quadrupole,
            parallel=parallel,
        )
        return {k: np.take(v, unsort, axis=0) for k, v in out.items()}

    def potential(self, pos_target=None, **kwargs):
        """Gravitational potential, shape (M,). See :meth:`evaluate`."""
        return self.evaluate(potential=True, pos_target=pos_target, **kwargs)["potential"]

    def accel(self, pos_target=None, **kwargs):
        """Gravitational acceleration, shape (M,3). See :meth:`evaluate`."""
        return self.evaluate(accel=True, pos_target=pos_target, **kwargs)["accel"]

    def tidal(self, pos_target=None, **kwargs):
        """Tidal tensor, shape (M,3,3). See :meth:`evaluate` and :func:`TidalTensor`."""
        return self.evaluate(tidal=True, pos_target=pos_target, **kwargs)["tidal"]


def DensityCorrFunc(
    pos,
    m,
    rbins=None,
    max_bin_size_ratio=100,
    theta=1.0,
    tree=None,
    return_tree=False,
    parallel=False,
    boxsize=0,
    weighted_binning=False,
):
    """Computes the average amount of mass in radial bin [r,r+dr] around a point, provided a set of radial bins.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    rbins: array_like or None, optional
        1D array of radial bin edges - if None will use heuristics to determine sensible bins. Otherwise MUST BE LOGARITHMICALLY SPACED (default None)
    max_bin_size_ratio: float, optional
        controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta: float, optional
        cell opening angle used to control accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0)
    parallel: bool, optional
        If True, will parallelize the correlation function computation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        if True will return the generated or used tree for future use (default False)
    boxsize: float, optional
        finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning: bool, optional
        (experimental) if True will distribute mass among radial bings with a weighted kernel (default False)

    Returns
    -------
    rbins: array_like
        array containing radial bin edges
    mbins: array_like
        array containing mean mass in radial bins, averaged over all points
    """

    if rbins is None:
        r = np.sort(np.sqrt(np.sum((pos - np.median(pos, axis=0)) ** 2, axis=1)))
        rbins = 10 ** np.linspace(np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3)))

    built_tree = tree is None
    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(_f64(pos), _f64(m), _f64(softening))  # build the tree if needed
    # Sort pos rather than applying the tree's stored permutation -- see the note in Potential. The
    # output is a binned aggregate, so sigma^2 was still a valid sample; the real failure was that a
    # supplied tree of a different size indexed out of bounds.
    idx = tree.TreewalkIndices if built_tree else _morton_order(_f64(pos))
    pos_sorted = np.take(pos, idx, axis=0)

    if parallel:
        mbins = DensityCorrFunc_tree_parallel(
            pos_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )
    else:
        mbins = DensityCorrFunc_tree(
            pos_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )

    if return_tree:
        return rbins, mbins, tree
    else:
        return rbins, mbins


def VelocityCorrFunc(
    pos,
    m,
    v,
    rbins=None,
    max_bin_size_ratio=100,
    theta=1.0,
    tree=None,
    return_tree=False,
    parallel=False,
    boxsize=0,
    weighted_binning=False,
):
    """Computes the weighted average product v(x).v(x+r), for a vector field v, in radial bins

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    v: array_like
        shape (N,3) of vector quantity (e.g. velocity, magnetic field, etc)
    rbins: array_like or None, optional
        1D array of radial bin edges - if None will use heuristics to determine sensible bins. Otherwise MUST BE LOGARITHMICALLY SPACED (default None)
    max_bin_size_ratio: float, optional
        controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta: float, optional
        cell opening angle used to control accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0)
    parallel: bool, optional
        If True, will parallelize the correlation function computation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        if True will return the generated or used tree for future use (default False)
    boxsize: float, optional
        finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning: bool, optional
        (experimental) if True will distribute mass among radial bings with a weighted kernel (default False)

    Returns
    -------
    rbins: array_like
        array containing radial bin edges
    corr: array_like
        array containing correlation function values in radial bins
    """

    if rbins is None:
        r = np.sort(np.sqrt(np.sum((pos - np.median(pos, axis=0)) ** 2, axis=1)))
        rbins = 10 ** np.linspace(np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3)))

    built_tree = tree is None
    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(_f64(pos), _f64(m), _f64(softening), vel=v)  # build the tree if needed
    # Sort pos rather than applying the tree's stored permutation -- see the note in Potential. The
    # output is a binned aggregate, so sigma^2 was still a valid sample; the real failure was that a
    # supplied tree of a different size indexed out of bounds.
    idx = tree.TreewalkIndices if built_tree else _morton_order(_f64(pos))
    pos_sorted = np.take(pos, idx, axis=0)
    v_sorted = np.take(v, idx, axis=0)
    wt_sorted = np.take(m, idx, axis=0)
    if parallel:
        corr = VelocityCorrFunc_tree_parallel(
            pos_sorted,
            v_sorted,
            wt_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )
    else:
        corr = VelocityCorrFunc_tree(
            pos_sorted,
            v_sorted,
            wt_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )

    if return_tree:
        return rbins, corr, tree
    else:
        return rbins, corr


def VelocityStructFunc(
    pos,
    m,
    v,
    rbins=None,
    max_bin_size_ratio=100,
    theta=1.0,
    tree=None,
    return_tree=False,
    parallel=False,
    boxsize=0,
    weighted_binning=False,
):
    """Computes the structure function for a vector field: the average value of (v(x) - v(x+r))^2, in radial bins for r

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    v: array_like
        shape (N,3) of vector quantity (e.g. velocity, magnetic field, etc)
    rbins: array_like or None, optional
        1D array of radial bin edges - if None will use heuristics to determine sensible bins. Otherwise MUST BE LOGARITHMICALLY SPACED (default None)
    max_bin_size_ratio: float, optional
        controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta: float, optional
        cell opening angle used to control accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0)
    parallel: bool, optional
        If True, will parallelize the correlation function computation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        if True will return the generated or used tree for future use (default False)
    boxsize: float, optional
        finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning: bool, optional
        (experimental) if True will distribute mass among radial bings with a weighted kernel (default False)

    Returns
    -------
    rbins: array_like
        array containing radial bin edges
    Sv: array_like
        array containing structure function values
    """

    if rbins is None:
        r = np.sort(np.sqrt(np.sum((pos - np.median(pos, axis=0)) ** 2, axis=1)))
        rbins = 10 ** np.linspace(np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3)))

    built_tree = tree is None
    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(_f64(pos), _f64(m), _f64(softening), vel=v)  # build the tree if needed
    # Sort pos rather than applying the tree's stored permutation -- see the note in Potential. The
    # output is a binned aggregate, so sigma^2 was still a valid sample; the real failure was that a
    # supplied tree of a different size indexed out of bounds.
    idx = tree.TreewalkIndices if built_tree else _morton_order(_f64(pos))
    pos_sorted = np.take(pos, idx, axis=0)
    v_sorted = np.take(v, idx, axis=0)
    wt_sorted = np.take(m, idx, axis=0)
    if parallel:
        Sv = VelocityStructFunc_tree_parallel(
            pos_sorted,
            v_sorted,
            wt_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )
    else:
        Sv = VelocityStructFunc_tree(
            pos_sorted,
            v_sorted,
            wt_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )

    if return_tree:
        return rbins, Sv, tree
    else:
        return rbins, Sv


def ColumnDensity(
    pos,
    m,
    radii,
    rays=None,
    randomize_rays=False,
    healpix=False,
    tree=None,
    theta=0.5,
    return_tree=False,
    parallel=False,
    group_size=16,
    device="cpu",
):
    """Ray-traced or angle-binned column density calculation.

    Returns an estimate of the column density from the position of each particle integrated to infinity, assuming the particles are represented by uniform spheres. Note that optical depth can be obtained by supplying "sigma = opacity * mass" in place of mass, useful in situations where opacity is highly variable.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    radii: array_like
        shape (N,) array containing particle radii of the uniform spheres that we use to model the particles' mass distribution
    rays: None, int, or array_like, optional
        Which ray directions to raytrace the columns (default None). None uses the angular-binned method with 6 bins on the sky; an integer uses that many rays, with 6 giving the standard 6-ray grid and other numbers sampling random directions; a (N_rays,3) array gives the directions explicitly, and is normalized automatically.
    healpix: int, optional
        If nonzero, use a healpix ray grid with this resolution level NSIDE (default False)
    randomize_rays: bool, optional
        Randomize the orientation of the ray-grid *for each particle* (default False)
    parallel: bool, optional
        If True, will parallelize the column density over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    theta: float, optional
        Opening angle for the beam-traced angular bin estimator (default 0.5)
    return_tree: bool, optional
        return the tree used for future use (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the descent (default 16). Bit-identical to the per-target walk for the ray-traced path; for rays=None the group opens a superset of nodes, so the answer shifts slightly and is somewhat more accurate. Ignored with randomize_rays or below COLUMN_GROUP_MIN_TARGETS targets; 1 gives the per-target walk.
    device: str, optional
        'cpu' (default) or 'cuda'. 'cuda' needs pytreegrav[cuda] and an NVIDIA GPU, and applies only to the ray-traced path (rays given, randomize_rays off). It is float32: relative error against this path is ~2e-6 typical, reaching ~1e-2 on the densest sightlines. It repacks and uploads the tree on every call -- for repeated evaluation hold a pytreegrav.cuda.CudaColumnDensity instead.

    Returns
    -------
    columns: array_like
        shape (N,N_rays) float array of column densities from particle centers integrated along the rays
    """

    if np.any(np.asarray(radii) <= 0):
        # Point-like, so no cross-section and nothing obscured -- the right h -> 0 limit, but it
        # drops their mass from the column, which should not pass unnoticed.
        warnings.warn(
            "Some particle radii are <= 0; these particles are point-like and contribute no column "
            "density. Supply a nonzero radius if their mass should be included."
        )

    pos = _f64(pos)
    if tree is None:
        tree = ConstructTree(
            pos,
            _f64(m),
            _f64(radii),
        )  # build the tree if needed
        # the tree was just built from pos, so its walk order already is pos's Morton order
        idx = tree.TreewalkIndices
    else:
        # A supplied tree may hold different particles than the targets, so its TreewalkIndices do
        # not index pos.  Sort the targets on their own, as PotentialTarget/AccelTarget do.
        idx = _morton_order(pos)
    pos_sorted = pos[idx]

    if type(rays) == int:
        if rays == 6:
            rays = np.vstack([np.eye(3), -np.eye(3)])  # 6-ray grid
        else:
            # generate a random grid of ray directions
            rays = np.random.normal(size=(rays, 3))  # normalize later
    elif type(rays) == np.ndarray:
        # check that the shape is correct
        if not len(rays.shape) == 2:
            raise Exception("rays array argument must be 2D.")
        elif rays.shape[1] != 3:
            raise Exception("rays array argument is not an array of 3D vectors.")
        rays = np.copy(rays)  # so that we don't overwrite the argument
    elif rays is not None:
        raise Exception("rays argument type is not supported")

    if healpix:
        # Imported here, not at module scope: healpy is not in install_requires (nothing else in the
        # package needs it), and `hp` was simply never bound, so this path raised NameError.
        try:
            import healpy as hp
        except ImportError as e:
            raise ImportError("healpix=... needs healpy: pip install healpy") from e
        nside = healpix
        npix = hp.nside2npix(nside)
        rays = np.array(hp.pix2vec(nside, np.arange(npix))).T

    if rays is not None:
        rays /= np.sqrt((rays * rays).sum(1))[:, None]  # normalize the ray vectors

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda":
        if rays is None or randomize_rays:
            raise ValueError("device='cuda' supports only the ray-traced path: pass rays, and not randomize_rays")
        from .cuda import CudaColumnDensity  # imported lazily; numba-cuda is an optional extra

        columns = CudaColumnDensity(tree)(pos_sorted, rays)
    # Grouping needs every target in a group to share the same ray directions, so randomize_rays --
    # which rotates the grid per target -- keeps the per-target walks.  Dispatched here rather than
    # inside ColumnDensity_tree: calling one parallel=True kernel from another serializes its prange.
    elif rays is not None and not randomize_rays and len(pos_sorted) >= COLUMN_GROUP_MIN_TARGETS:
        walk = ColumnDensity_grouped_parallel if parallel else ColumnDensity_grouped
        columns = walk(pos_sorted, rays, tree, group_size)
    elif rays is None and len(pos_sorted) >= COLUMN_GROUP_MIN_TARGETS:
        # Not merely faster: opening a superset of nodes splits each node's mass across the sky bins
        # more finely, roughly halving the bin-to-bin scatter. group_size=1 recovers the old result.
        walk = ColumnDensityBinned_grouped_parallel if parallel else ColumnDensityBinned_grouped
        columns = walk(pos_sorted, tree, theta, group_size)
    elif parallel:
        columns = ColumnDensity_tree_parallel(pos_sorted, tree, rays, randomize_rays=randomize_rays, theta=theta)
    else:
        columns = ColumnDensity_tree(pos_sorted, tree, rays, randomize_rays=randomize_rays, theta=theta)
    if np.any(np.isnan(columns)):
        print("WARNING some column densities are NaN!")
    unsorted = np.empty_like(columns)
    unsorted[idx] = columns  # undo the permutation
    columns = unsorted

    if return_tree:
        return columns, tree
    else:
        return columns
