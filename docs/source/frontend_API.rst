API Documentation
=================

.. _potential-options:

Potential options
-----------------

``Potential`` and ``PotentialTarget`` accept the keyword-only options
``softening_kernel="cubic_spline"`` and ``self_index=None``. ``Field`` fixes the
kernel at construction and accepts ``self_index`` in ``potential`` or
potential-only ``evaluate`` calls. The defaults preserve existing behavior.
The :doc:`quickstart <usage/quickstart>` includes examples with independent
targets, source subsets, and reusable fields.

Softening, normalization, and units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Softening replaces a point source by a normalized spherical density. The input
support radius :math:`H` is a length: gravity becomes exactly Newtonian outside
it. For ``softening_kernel="wendland_c2"`` and :math:`H>0`, the unit-mass density is

.. math::

   W(r,H) = \frac{21}{2\pi H^3}(1-q)^4(1+4q),\qquad q=r/H,\quad 0\le q<1,

and zero outside. Its spherical integral is one. Integrating the enclosed mass
and setting the potential to zero at infinity gives the unit-mass potential

.. math::

   \psi(r,H) =
   \begin{cases}
   (3q^7 - 15q^6 + 28q^5 - 21q^4 + 7q^2 - 3)/H, & 0\le r<H,\ H>0,\\
   -1/r, & r\ge H,\ r>0.
   \end{cases}

In particular, :math:`\psi(0,H)=-3/H`, :math:`\psi(H/2,H)=-243/(128H)`, and
:math:`\psi(H,H)=-1/H`. The potential and its first two radial derivatives match
the exterior Newtonian expression at the boundary. Zero support at positive
separation gives the point-mass potential.

The returned specific potential is
:math:`\phi_i=G\sum_j m_j\psi(r_{ij},\max(H_i,H_j))`, with the selected
exclusions applied. For example, target support 0.2 and source support 0.5 use
pair support 0.5. Supply masses, positions, supports, and ``G`` in consistent
units; the result has units :math:`G\,\mathrm{mass}/\mathrm{length}`. There is
no automatic unit conversion and no factor of one half in ``phi``.

The compact support is not a Plummer softening. Matching the central potential
:math:`-Gm/\epsilon_{\rm Plummer}` requires :math:`H=3\epsilon_{\rm Plummer}`
for Wendland C2, or :math:`H=2.8\epsilon_{\rm Plummer}` for cubic spline. Equal
support radii therefore differ from equal Plummer-equivalent softenings. These
are gravitational conventions, not SPH smoothing-length conversions.

The Wendland convention follows the default gravity softening in the public
`SWIFT planetary archive <https://doi.org/10.5281/zenodo.15973312>`_, revision
``c1c602152b7745c15e0790940069a3d314781dfb``. The expression above is independently
derived from the normalized density; no upstream implementation is copied.

Identity and coincidence
^^^^^^^^^^^^^^^^^^^^^^^^

``self_index`` controls exclusions independently of the selected kernel:

* ``None`` preserves legacy rules. Self-direct ``Potential`` excludes each
  particle itself but includes other softened particles at the same position.
  Target-direct and tree evaluations, including ``Field``, omit all
  zero-separation pairs. Kernel selection alone preserves this discrepancy.
* An integer array of shape ``(N_target,)`` enables identity mode. Entry ``j``
  excludes original source ``j``; ``-1`` excludes no source. Distinct coincident
  sources contribute their finite central potential when pair support is
  positive. Repeated source indices are allowed for repeated queries.
* ``"self"`` means ``np.arange(N_source)`` and is accepted only by ``Potential``
  without a supplied tree, or by ``Field`` when ``pos_target=None``.

Mapping a source is an explicit claim of identity, checked by exact equality of
the target and stored source positions after float64 conversion. Coordinates
are never used to discover identities. Independent or displaced targets use
``-1``. Boolean/fractional arrays, wrong shapes, and out-of-range indices are
rejected. ``PotentialTarget`` does not accept the ``"self"`` shorthand.

With a supplied tree, the mapping refers to its **original input source order**,
not internal Morton order or target order. The stored positions, masses, and
supports are authoritative; arrays passed alongside it do not replace them.
Adaptive evaluation uses the tree, and ``method="bruteforce"`` with a tree is
rejected for these options. A new source subset tree has its own local indices;
translate external particle IDs into that namespace before calling. Results
always follow target order.

A tree or ``Field`` represents fixed sources. Rebuild after changing source
positions, masses, or supports; do not mutate its internal arrays. To change a
``Field`` kernel, construct another ``Field``. Self evaluation through ``Field``
uses the stored source supports and does not accept ``softening_target``.

In identity mode, a retained positive-mass source at zero separation with zero
pair support raises ``ValueError``. Mapped self contributions and zero-mass
sources are skipped before this check. Nonfinite output and overflow or
underflow of squared separations also raise, rather than silently returning a
zero or coincident-pair potential outside the supported numerical range.
When either new option is active, inputs require finite positions,
finite nonnegative masses/supports, finite positive ``G`` and ``theta``, and a
positive integer ``group_size``. Accepted numeric arrays are converted to
float64, and the output is float64.

Supported paths and accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Applies when Wendland or explicit identity is selected
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Support
   * - CPU ``Potential`` / ``PotentialTarget``
     - Serial/parallel direct and tree, in float64
   * - Static radix/Morton tree with moments
     - Reusable; monopoles or stored quadrupoles
   * - ``Field.potential`` / potential-only ``evaluate``
     - Reusable; grouped or ``group_size=1``
   * - Insertion/dynamic trees, spatial-only trees, unresolved distinct radix buckets
     - Rejected
   * - CUDA, acceleration/tidal fields, fused derivative requests
     - Unsupported; rejected rather than silently changing kernels
   * - Legacy low-level walks and symmetric direct entry points
     - No new option-bearing API; use the frontend functions

Optional tree evaluations open nodes until all represented source/target pairs
are outside their softening supports before using Newtonian multipoles. This
guard is not a bound on multipole error. Smaller ``theta`` generally improves
accuracy; grouping and quadrupoles also affect cost and error. Broad or dense
overlapping supports may approach direct-summation cost. Optional direct
self evaluations sum ordered pairs, bypassing the legacy triangular/symmetric
optimizations. Existing adaptive thresholds are heuristics, not guarantees of
the fastest method for these options. Warm up each Numba signature before
comparing runtimes and measure construction separately from reused evaluation.

Choose an error budget for your quantity and data. Compare against direct
summation on a manageable population or representative sampled targets, using
the same kernel, supports, ``G``, and identity rules. Report absolute errors,
fractional errors with a stated near-zero floor, and tails as well as averages.
Check convergence under smaller ``theta``; a default opening angle or a force
error estimate does not establish a potential or energy tolerance. Differences
between independently approximated source subsets can amplify cancellation
error and need their own validation.

For a complete self-interacting population with explicit self exclusion, total
potential energy is :math:`U=\frac12\sum_i m_i\phi_i`. For two disjoint
populations, their cross energy is :math:`\sum_{i\in A}m_i\phi_B(x_i)` without
the factor one half. Check mass-weighted errors and energy separately from
per-particle potential errors. Sampled target comparisons do not constitute an
exact full-population energy reference.

.. automodule:: pytreegrav.frontend
   :noindex:
   :members:

Direct summation
----------------

``Potential`` and ``Accel`` fall back to exact direct summation for small particle counts, and can be
forced to it with ``method="bruteforce"``. Two parallel implementations exist and the frontend picks
between them automatically for default calls:

* below ``SYMMETRIC_NMIN`` particles, the straightforward kernel, which gives each thread one target
  and therefore evaluates all :math:`N^2` pairs;
* at or above it, the symmetrized kernel below, which evaluates each pair once -- half the flops --
  at the cost of per-thread scratch buffers.

You normally do not need to call these directly; they are documented because the crossover and the
memory cost are worth knowing about. Selecting Wendland or explicit identity uses a separate
target-sum implementation instead.

.. automodule:: pytreegrav.bruteforce_symmetric
   :noindex:
   :members:

.. automodule:: pytreegrav.bruteforce
   :noindex:
   :members:
