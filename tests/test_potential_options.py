"""Optional potential kernels and explicit source identities, using synthetic data.

The Wendland reference integrates its normalized spherical density through the
shell theorem; it does not call or reproduce the implementation's potential
polynomial. No external datasets or reference packages are needed.
"""

import warnings

import numpy as np
import pytest

import pytreegrav.frontend as frontend
from pytreegrav import Accel, ConstructTree, Field, Potential, PotentialTarget, TidalTensor


WENDLAND = "wendland_c2"
_GX, _GW = np.polynomial.legendre.leggauss(12)


def _integrate(function, lo, hi):
    x = lo + (hi - lo) * (_GX + 1) / 2
    return (hi - lo) / 2 * np.dot(_GW, function(x))


def _radial_mass(q):
    # 4*pi*r**2*rho(r) in dimensionless radius for unit support and mass.
    return 42 * q**2 * (1 - q)**4 * (1 + 4 * q)


def _wendland(r, support):
    if support == 0 or r >= support:
        return -1 / r
    q = r / support
    interior = _integrate(_radial_mass, 0, q) / q if q else 0
    shells = _integrate(lambda u: 42 * u * (1 - u)**4 * (1 + 4 * u), q, 1)
    return -(interior + shells) / support


def _reference(target, source, masses, ht, hs, indices, G=1):
    return np.array([
        G * sum(
            m * _wendland(np.linalg.norm(x - y), max(h, s))
            for j, (y, m, s) in enumerate(zip(source, masses, hs))
            if j != excluded and m != 0
        )
        for x, h, excluded in zip(target, ht, indices)
    ])


def _cloud(n=32, seed=17):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.uniform(0.1, 2, n), rng.uniform(0.02, 0.7, n)


@pytest.mark.parametrize("parallel", [False, True])
def test_wendland_shell_theorem_and_exact_values(parallel):
    assert _integrate(_radial_mass, 0, 1) == pytest.approx(1, abs=2e-15)
    support, mass, G = 2.0, 1.7, 2.3
    q = np.array([0, 1e-12, 0.13, 0.5, 0.79, np.nextafter(1., 0.), 1, np.nextafter(1., 2.), 2])
    target = np.column_stack((support * q, np.zeros((len(q), 2))))
    got = PotentialTarget(
        target, [[0., 0., 0.]], [mass], softening_source=[support],
        G=G, method="bruteforce", parallel=parallel,
        softening_kernel=WENDLAND, self_index=np.full(len(q), -1),
    )
    expected = G * mass * np.array([_wendland(r, support) for r in support * q])
    np.testing.assert_allclose(got, expected, rtol=8e-15, atol=0)
    np.testing.assert_allclose(got[[0, 3, 6]] * support / (G * mass), [-3, -243 / 128, -1], rtol=2e-15)
    # Enclosed mass and the two exterior derivatives follow independently from
    # the density; centered differences check the implemented joining point.
    eps = 2e-4
    points = np.column_stack((support * np.array([1 - eps, 1, 1 + eps]), np.zeros((3, 2))))
    values = PotentialTarget(points, [[0., 0., 0.]], [1.], softening_source=[support],
                             method="bruteforce", softening_kernel=WENDLAND)
    first = (values[2] - values[0]) / (2 * support * eps)
    second = (values[2] - 2 * values[1] + values[0]) / (support * eps)**2
    assert first == pytest.approx(1 / support**2, rel=2e-6)
    assert second == pytest.approx(-2 / support**3, rel=2e-6)


@pytest.mark.parametrize("kernel, central", [("cubic_spline", 2.8), (WENDLAND, 3.)])
@pytest.mark.parametrize("method, parallel", [("bruteforce", False), ("bruteforce", True), ("tree", False), ("tree", True)])
def test_coincident_identities_pair_support_and_energy(kernel, central, method, parallel):
    x = np.zeros((2, 3))
    m, h, G = np.array([2., 5.]), np.array([0., 0.7]), 1.9
    options = dict(softening_kernel=kernel, method=method, parallel=parallel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        phi = Potential(x, m, h, G=G, self_index="self", **options)
        independent = PotentialTarget(x[:1], x, m, [h[1]], h, G=G, self_index=[-1], **options)
    np.testing.assert_allclose(phi, -G * central / h[1] * m[::-1], rtol=2e-15)
    np.testing.assert_allclose(independent, [-G * central * m.sum() / h[1]], rtol=2e-15)
    assert 0.5 * np.dot(m, phi) == pytest.approx(-G * central * np.prod(m) / h[1])


@pytest.mark.parametrize("method", ["bruteforce", "tree"])
@pytest.mark.parametrize("kernel, central", [("cubic_spline", 2.8), (WENDLAND, 3.)])
def test_single_identity_and_independent_central_target(method, kernel, central):
    options = dict(method=method, softening_kernel=kernel)
    assert Potential([[0., 0., 0.]], [2.], self_index="self", **options)[0] == 0
    got = PotentialTarget([[0., 0., 0.]], [[0., 0., 0.]], [2.],
                          softening_source=[0.5], self_index=[-1], **options)
    assert got[0] == pytest.approx(-central * 2 / 0.5)


@pytest.mark.parametrize("method, parallel", [("bruteforce", False), ("bruteforce", True), ("tree", False), ("tree", True)])
def test_singular_pairs_raise_but_zero_mass_and_mapped_self_do_not(method, parallel):
    options = dict(method=method, parallel=parallel, softening_kernel=WENDLAND)
    with pytest.raises(ValueError, match="nonfinite potential"):
        PotentialTarget([[0., 0., 0.]], [[0., 0., 0.]], [1.], self_index=[-1], **options)
    for masses in ([0.], [2.]):
        assert Potential([[0., 0., 0.]], masses, self_index="self", **options)[0] == 0
    assert PotentialTarget([[0., 0., 0.]], [[0., 0., 0.]], [0.], self_index=[-1], **options)[0] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="nonfinite potential"):
            Potential(np.zeros((2, 3)), [1., 2.], self_index="self", **options)


@pytest.mark.parametrize("separation", [1e200, 1e-200])
@pytest.mark.parametrize("kernel, indices", [(WENDLAND, [-1]), (WENDLAND, None), ("cubic_spline", [-1])])
@pytest.mark.parametrize("method, parallel", [("bruteforce", False), ("bruteforce", True), ("tree", False), ("tree", True)])
def test_unrepresentable_squared_distance_raises_instead_of_losing_a_pair(separation, kernel, indices, method, parallel):
    # These finite inputs give exactly -m/r = -1. Naively squaring their
    # separation produces infinity or zero, silently losing a contribution
    # or turning distinct particles into a false coincidence.
    with pytest.raises(ValueError, match="nonfinite potential"):
        PotentialTarget(
            [[separation, 0., 0.]], [[0., 0., 0.]], [separation],
            method=method, parallel=parallel, softening_kernel=kernel, self_index=indices,
        )


def test_kernel_choice_preserves_legacy_coincidence_policies():
    x, m, h = np.zeros((2, 3)), np.array([2., 5.]), np.ones(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        np.testing.assert_allclose(Potential(x, m, h, method="bruteforce", softening_kernel=WENDLAND), [-15, -6])
        np.testing.assert_array_equal(Potential(x, m, h, method="tree", softening_kernel=WENDLAND), [0, 0])
        for method in ("tree", "bruteforce"):
            np.testing.assert_array_equal(PotentialTarget(x, x, m, h, h, method=method,
                                                         softening_kernel=WENDLAND), [0, 0])


@pytest.mark.parametrize("quadrupole, parallel, group_size", [
    (False, False, 1), (False, True, 8), (True, False, 8), (True, True, 1),
])
def test_tree_field_permutations_reuse_and_input_preservation(quadrupole, parallel, group_size):
    x, m, h = _cloud(34)
    x[:10] = x[0]  # longer than one eight-child coincidence bucket
    # Force a second Morton-key resolution pass without exhausting its budget.
    x[12:20] = 0.5 + 1e-8 * x[12:20]
    rng = np.random.default_rng(4)
    source_order = rng.permutation(len(x))
    x, m, h = x[source_order], m[source_order], h[source_order]
    target_ids = np.array([21, 0, 5, 21, 33, 8])
    target = np.vstack((x[target_ids], [[3., -1., 2.]], x[[4]]))
    ids = np.r_[target_ids, -1, -1]
    ht = np.r_[h[target_ids], 0.2, 0.8]
    # Strided caller-owned inputs, including a map, must retain their values.
    arrays = []
    for a in (x, m, h, target, ht, ids):
        buffer = np.empty((2 * len(a),) + a.shape[1:], dtype=a.dtype)
        buffer[::2] = a
        arrays.append(buffer[::2])
    x, m, h, target, ht, ids = arrays
    snapshots = [a.copy() for a in arrays]
    expected = _reference(target, x, m, ht, h, ids, G=1.7)
    controls = dict(G=1.7, theta=1e-8, quadrupole=quadrupole, parallel=parallel, group_size=group_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        got, tree = PotentialTarget(target, x, m, ht, h, method="tree", return_tree=True,
                                    softening_kernel=WENDLAND, self_index=ids, **controls)
        field = Field(x, m, h, softening_kernel=WENDLAND, **controls)
    np.testing.assert_allclose(got, expected, rtol=8e-15)
    for repeat in range(2):
        reused = PotentialTarget(target, None, None, softening_target=ht, tree=tree,
                                 softening_kernel=WENDLAND, self_index=ids, **controls)
        cached = field.potential(target, softening_target=ht, self_index=ids)
        np.testing.assert_array_equal(reused, got)
        np.testing.assert_array_equal(cached, got)
    self_expected = _reference(x, x, m, h, h, np.arange(len(x)), G=1.7)
    np.testing.assert_allclose(field.potential(self_index="self"), self_expected, rtol=8e-15)
    np.testing.assert_allclose(Potential(x, m, h, method="bruteforce", G=1.7,
                                        softening_kernel=WENDLAND, self_index="self"), self_expected, rtol=8e-15)
    for actual, before in zip(arrays, snapshots):
        np.testing.assert_array_equal(actual, before)


def test_source_reordering_requires_remapping_original_source_indices():
    x, m, h = _cloud(12)
    target_ids = np.array([8, 2, 9, 2])
    target = x[target_ids]
    reorder = np.array([9, 0, 5, 1, 2, 6, 8, 7, 4, 11, 10, 3])
    options = dict(method="tree", theta=1e-8, softening_kernel=WENDLAND)
    before = PotentialTarget(target, x, m, h[target_ids], h, self_index=target_ids, **options)
    after = PotentialTarget(target, x[reorder], m[reorder], h[target_ids], h[reorder],
                            self_index=np.argsort(reorder)[target_ids], **options)
    np.testing.assert_allclose(after, before, rtol=4e-15)
    with pytest.raises(ValueError, match="source positions"):
        PotentialTarget(target, x[reorder], m[reorder], h[target_ids], h[reorder],
                        self_index=target_ids, **options)


def test_subset_source_partition_and_cross_energy():
    x, m, h = _cloud(18)
    x[3] = x[0]  # a retained distinct particle, even across source partitions
    targets = np.array([0, 7, 3, 14])
    options = dict(method="bruteforce", softening_kernel=WENDLAND, G=1.4)
    full = PotentialTarget(x[targets], x, m, h[targets], h, self_index=targets, **options)
    parts = []
    for source_ids in (np.arange(0, len(x), 2), np.arange(1, len(x), 2)):
        local_ids = {original: local for local, original in enumerate(source_ids)}
        indices = np.array([local_ids.get(original, -1) for original in targets])
        part = PotentialTarget(x[targets], x[source_ids], m[source_ids], h[targets], h[source_ids],
                               self_index=indices, **options)
        expected = _reference(x[targets], x[source_ids], m[source_ids], h[targets], h[source_ids], indices, G=1.4)
        np.testing.assert_allclose(part, expected, rtol=4e-15)
        parts.append(part)
    np.testing.assert_allclose(full - parts[0], parts[1], rtol=4e-15)
    # Cross energy between disjoint populations uses each pair once, without
    # the factor one-half required for a complete self-energy sum.
    left, right = np.arange(0, len(x), 2), np.arange(1, len(x), 2)
    cross = PotentialTarget(x[left], x[right], m[right], h[left], h[right],
                            self_index=np.full(len(left), -1), **options)
    expected_energy = sum(
        1.4 * m[i] * m[j] * _wendland(np.linalg.norm(x[i] - x[j]), max(h[i], h[j]))
        for i in left for j in right
    )
    assert np.dot(m[left], cross) == pytest.approx(expected_energy, rel=4e-15)


@pytest.mark.parametrize("quadrupole", [False, True])
def test_cube_half_diagonal_opens_node_with_a_softened_corner_pair(quadrupole):
    x = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    target, h = np.array([[0.56, 0.56, 0.56]]), np.full(2, 0.3)
    tree = ConstructTree(x, np.ones(2), h, quadrupole=quadrupole)
    root = tree.NumParticles
    distance = np.linalg.norm(target[0] - tree.Coordinates[root])
    size, delta = tree.Sizes[root], tree.Deltas[root]
    assert distance > max(size / 2 + delta, 0.3 + 0.6 * size + delta)
    assert distance < 0.3 + np.sqrt(3) / 2 * size + delta
    expected = _reference(target, x, np.ones(2), [0.], h, [-1])
    got = PotentialTarget(target, None, None, tree=tree, theta=2, group_size=1,
                          quadrupole=quadrupole, softening_kernel=WENDLAND, self_index=[-1])
    np.testing.assert_allclose(got, expected, rtol=3e-15)


def test_realistic_opening_accuracy_scaling_and_newtonian_limit():
    x, m, h = _cloud(256)
    target, _, ht = _cloud(21, seed=41)
    ids = np.full(len(target), -1)
    expected = _reference(target, x, m, ht, h, ids)
    for quadrupole in (False, True):
        got = PotentialTarget(target, x, m, ht, h, theta=0.5, group_size=1,
                              method="tree", quadrupole=quadrupole,
                              softening_kernel=WENDLAND, self_index=ids)
        np.testing.assert_allclose(got, expected, rtol=3e-3)
    scaled = PotentialTarget(7 * target, 7 * x, 11 * m, 7 * ht, 7 * h,
                             G=3, method="bruteforce", softening_kernel=WENDLAND, self_index=ids)
    np.testing.assert_allclose(scaled, expected * 33 / 7, rtol=6e-15)
    got = PotentialTarget([[0., 0., 0.]], [[2., 0., 0.]], [3.],
                          softening_kernel=WENDLAND, self_index=[-1])
    np.testing.assert_array_equal(got, [-1.5])


@pytest.mark.parametrize("indices", [[True], [0.], [[0]], [0, 0], [-2], [1], np.array([2**64 - 1], dtype=np.uint64), "self"])
def test_invalid_identity_maps(indices):
    with pytest.raises(ValueError, match="self_index"):
        PotentialTarget([[0., 0., 0.]], [[0., 0., 0.]], [1.], self_index=indices)


@pytest.mark.parametrize("kernel", ["wendland", "", None, 1, ["wendland_c2"]])
def test_invalid_kernels(kernel):
    with pytest.raises(ValueError, match="softening_kernel"):
        Potential([[0., 0., 0.]], [1.], softening_kernel=kernel)
    with pytest.raises(ValueError, match="softening_kernel"):
        Field([[0., 0., 0.]], [1.], softening_kernel=kernel)


@pytest.mark.parametrize("override", [
    {"pos_source": [[np.nan, 0, 0]]}, {"pos_target": [[0, 0]]},
    {"m_source": [-1.]}, {"m_source": [np.inf]}, {"m_source": [1, 2]},
    {"softening_source": [-1.]}, {"softening_source": [np.nan]}, {"softening_target": [1, 2]},
    {"G": 0}, {"G": -1}, {"G": np.inf}, {"theta": 0}, {"theta": np.nan},
    {"group_size": 0}, {"group_size": True}, {"group_size": 1.5},
    {"pos_source": None}, {"m_source": None}, {"method": "invalid"},
])
def test_invalid_optional_inputs(override):
    kwargs = dict(pos_target=[[1., 0., 0.]], pos_source=[[0., 0., 0.]], m_source=[1.],
                  softening_kernel=WENDLAND)
    kwargs.update(override)
    with pytest.raises(ValueError):
        PotentialTarget(**kwargs)


def test_empty_inputs_and_float64_coercion():
    empty = np.empty((0, 3))
    for method in ("bruteforce", "tree"):
        result = PotentialTarget(empty, [[0., 0., 0.]], [1.], method=method,
                                 softening_kernel=WENDLAND, self_index=np.empty(0, dtype=int))
        assert result.shape == (0,)
    got = PotentialTarget([[1., 0., 0.]], empty, [], method="bruteforce", softening_kernel=WENDLAND)
    np.testing.assert_array_equal(got, [0])
    with pytest.raises(ValueError, match="at least one source"):
        PotentialTarget([[1., 0., 0.]], empty, [], method="tree", softening_kernel=WENDLAND)
    with pytest.raises(ValueError, match="at least one source"):
        Field(empty, [], softening_kernel=WENDLAND)
    result = Potential(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
                       np.ones(2, np.float32), self_index="self", softening_kernel=WENDLAND)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [-1, -1])


@pytest.mark.parametrize("construction", [{"radix": False}, {"morton_order": False},
                                          {"compute_moments": False}, {"vel": np.zeros((3, 3))}])
def test_unsupported_tree_builders(construction):
    x, m, h = _cloud(3)
    tree = ConstructTree(x, m, h, **construction)
    with pytest.raises(ValueError, match="static radix/Morton tree with moments"):
        PotentialTarget(x, None, None, tree=tree, softening_kernel=WENDLAND)


def test_unresolved_distinct_bucket_is_rejected():
    x = np.array([[0., 0., 0.], [1e-100, 0., 0.], [1., 1., 1.]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tree = ConstructTree(x, np.ones(3), np.ones(3))
    assert tree.HasUnresolvedPoints
    with pytest.raises(ValueError, match="unresolved distinct"):
        PotentialTarget(x, None, None, tree=tree, softening_kernel=WENDLAND)


def test_reused_tree_is_authoritative_and_requires_explicit_mapping():
    x, m, h = _cloud(16)
    options = dict(softening_kernel=WENDLAND, theta=1e-8, self_index=np.arange(len(x)))
    tree = ConstructTree(x, m, h)
    expected = PotentialTarget(x, None, None, softening_target=h, tree=tree, **options)
    # Supplied source arrays must not replace the stored source data or select
    # direct summation merely because their population is small.
    got = PotentialTarget(x, x + 1, 9 * m, h, 2 * h, tree=tree, **options)
    np.testing.assert_array_equal(got, expected)
    with pytest.raises(ValueError, match="supplied tree"):
        PotentialTarget(x, x, m, tree=tree, method="bruteforce", **options)
    with pytest.raises(ValueError, match="known source order"):
        Potential(x, m, h, tree=tree, softening_kernel=WENDLAND, self_index="self")
    with pytest.raises(ValueError, match="quadrupole moments"):
        PotentialTarget(x, None, None, tree=tree, quadrupole=True, **options)
    with pytest.raises(ValueError, match="source positions"):
        PotentialTarget(x + 1e-6, None, None, tree=tree, **options)


@pytest.mark.parametrize("kernel, indices", [(WENDLAND, None), ("cubic_spline", "self")])
def test_field_rejects_derivative_requests(kernel, indices):
    x, m, h = _cloud(4)
    field = Field(x, m, h, softening_kernel=kernel)
    for flags in (dict(accel=True), dict(tidal=True), dict(potential=True, accel=True), dict(potential=True, tidal=True)):
        with pytest.raises(ValueError, match="potential-only"):
            field.evaluate(self_index=indices, **flags)
    with pytest.raises(ValueError, match="stored source softenings"):
        field.potential(self_index="self", softening_target=h)


def test_other_backends_reject_new_options():
    x, m, h = _cloud(3)
    with pytest.raises(ValueError, match="cpu"):
        Potential(x, m, h, device="cuda", softening_kernel=WENDLAND)
    for function in (Accel, TidalTensor):
        with pytest.raises(TypeError):
            function(x, m, h, softening_kernel=WENDLAND)


@pytest.mark.parametrize("parallel, n, expected", [
    (False, 1000, "direct"), (False, 1001, "tree"),
    (True, 4000, "direct"), (True, 4001, "tree"),
])
def test_self_adaptive_dispatch_without_large_pair_sums(monkeypatch, parallel, n, expected):
    calls = []
    def direct(pos, *args):
        calls.append("direct")
        return np.zeros(len(pos))
    def tree_values(pos, *args, **kwargs):
        calls.append("tree")
        return np.zeros(len(pos))
    monkeypatch.setattr(frontend, "_PotentialTarget_options_serial", direct)
    monkeypatch.setattr(frontend, "_PotentialTarget_options_parallel", direct)
    monkeypatch.setattr(frontend, "ConstructTree", lambda *a, **k: object())
    monkeypatch.setattr(frontend, "_check_potential_tree", lambda *a: None)
    monkeypatch.setattr(frontend, "_potential_tree_values", tree_values)
    monkeypatch.setattr(frontend, "Potential_bruteforce_symmetric", lambda *a, **k: pytest.fail("legacy symmetric dispatch"))
    x, m, h = _cloud(n)
    Potential(x, m, h, parallel=parallel, softening_kernel=WENDLAND, self_index="self")
    assert calls == [expected]


@pytest.mark.parametrize("n_target, expected", [(999, "direct"), (1000, "direct"), (1001, "tree")])
def test_target_adaptive_work_threshold(monkeypatch, n_target, expected):
    calls = []
    def direct(pos, *args):
        calls.append("direct")
        return np.zeros(len(pos))
    def tree_values(pos, *args, **kwargs):
        calls.append("tree")
        return np.zeros(len(pos))
    monkeypatch.setattr(frontend, "_PotentialTarget_options_serial", direct)
    monkeypatch.setattr(frontend, "ConstructTree", lambda *a, **k: object())
    monkeypatch.setattr(frontend, "_check_potential_tree", lambda *a: None)
    monkeypatch.setattr(frontend, "_potential_tree_values", tree_values)
    x, m, h = _cloud(1000)
    target, _, ht = _cloud(n_target, seed=99)
    PotentialTarget(target, x, m, ht, h, softening_kernel=WENDLAND)
    assert calls == [expected]
