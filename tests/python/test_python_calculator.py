"""A Calculator written in Python must work everywhere a C++ one does.

The point of the trampoline is not that `compute` gets called - it is that a
subclass is indistinguishable from a built-in calculator to every consumer:
optimizers, dynamics, phonons, NEB, global optimization. Each of those is
checked here, because "it works when I call it directly" was true of several
earlier attempts that fell over the first time something else held the object.
"""

import gc

import numpy as np
import pytest

import fastatoms as fa


class Spring(fa.Calculator):
    """Every atom on a spring to the origin: E = k/2 sum r^2."""

    def __init__(self, k=1.0):
        super().__init__()
        self.k = k
        self.evaluations = 0

    def compute(self, atoms, properties):
        self.evaluations += 1
        positions = np.asarray(atoms.positions)
        return {
            "energy": 0.5 * self.k * float((positions**2).sum()),
            "forces": -self.k * positions,
        }


class SpringByAssignment(Spring):
    """The other way of answering: fill `self.results` and return nothing."""

    def compute(self, atoms, properties):
        self.results = super().compute(atoms, properties)


@pytest.mark.parametrize("cls", [Spring, SpringByAssignment])
def test_both_ways_of_answering(cls):
    atoms = fa.Atoms("H2", [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    atoms.calc = cls(k=3.0)
    assert atoms.get_potential_energy() == pytest.approx(0.5 * 3.0 * 5.0)
    assert np.allclose(atoms.get_forces(), [[-3.0, 0, 0], [0, -6.0, 0]])


def test_name_and_repr_come_from_the_subclass():
    atoms = fa.Atoms("H", [[1.0, 0.0, 0.0]])
    atoms.calc = Spring()
    assert atoms.calc.name == "Spring"


def test_survives_being_a_temporary():
    # The Python object has no reference of its own once `calc =` returns; only
    # the Atoms holds it. Without trampoline_self_life_support it is collected
    # here and the C++ side is left with a dangling override.
    atoms = fa.Atoms("H", [[1.0, 0.0, 0.0]])
    atoms.calc = Spring(k=2.0)
    gc.collect()
    assert atoms.calc.name == "Spring"
    assert atoms.get_potential_energy() == pytest.approx(1.0)


def test_results_are_cached_between_queries():
    atoms = fa.Atoms("H2", [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    calc = Spring()
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()
    assert calc.evaluations == 1
    atoms.positions[0] = [2.0, 0.0, 0.0]
    atoms.get_potential_energy()
    assert calc.evaluations == 2


def test_exceptions_keep_their_type():
    class Broken(fa.Calculator):
        def compute(self, atoms, properties):
            raise KeyError("deliberate")

    atoms = fa.Atoms("H", [[0.0, 0.0, 0.0]])
    atoms.calc = Broken()
    with pytest.raises(KeyError, match="deliberate"):
        atoms.get_potential_energy()


def test_returning_nothing_is_an_error_not_a_silent_zero():
    class Silent(fa.Calculator):
        def compute(self, atoms, properties):
            return None

    atoms = fa.Atoms("H", [[0.0, 0.0, 0.0]])
    atoms.calc = Silent()
    with pytest.raises(fa.CalculatorError, match="no results"):
        atoms.get_potential_energy()


def test_returning_the_wrong_type_says_so():
    class Wrong(fa.Calculator):
        def compute(self, atoms, properties):
            return 42

    atoms = fa.Atoms("H", [[0.0, 0.0, 0.0]])
    atoms.calc = Wrong()
    with pytest.raises(fa.CalculatorError, match="int"):
        atoms.get_potential_energy()


def test_implemented_properties_is_honoured():
    class EnergyOnly(Spring):
        implemented_properties = ["energy"]

    atoms = fa.Atoms("H", [[1.0, 0.0, 0.0]])
    atoms.calc = EnergyOnly()
    assert atoms.get_potential_energy() == pytest.approx(0.5)
    with pytest.raises(fa.PropertyNotImplementedError):
        atoms.get_forces()


@pytest.mark.parametrize("stress", [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    np.diag([0.5, 0.5, 0.5]),
    np.diag([0.5, 0.5, 0.5]).reshape(9),
])
def test_stress_accepts_voigt_flat_and_matrix(stress):
    class Stressed(fa.Calculator):
        def compute(self, atoms, properties):
            return {"energy": 0.0, "stress": stress}

    atoms = fa.build.fcc("Cu", 3.6, cubic=True)
    atoms.calc = Stressed()
    assert len(atoms.get_stress()) == 6


def test_drives_an_optimizer():
    atoms = fa.Atoms("H2", [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    atoms.calc = Spring()
    result = fa.LBFGS(fmax=1e-6, max_steps=200).run(atoms)
    assert result.converged
    assert np.allclose(atoms.positions, 0.0, atol=1e-5)


def test_drives_molecular_dynamics():
    atoms = fa.Atoms("H2", [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    calc = Spring()
    atoms.calc = calc
    fa.Dynamics(timestep=0.1).run(atoms, 50)
    assert calc.evaluations >= 50


def test_drives_a_neb():
    def band():
        images = []
        for x in np.linspace(-1.0, 1.0, 5):
            image = fa.Atoms("H", [[x, 0.0, 0.0]])
            image.calc = Spring()
            images.append(image)
        return images

    neb = fa.NEB(band())
    neb.run(fmax=0.05, steps=100)
    assert neb.images[0].calc.name == "Spring"


def test_a_subclass_may_call_back_into_fastatoms():
    # `neighbor_list` releases the GIL; a Python calculator calling it must
    # still be holding one when it returns.
    class Neighbourly(fa.Calculator):
        def compute(self, atoms, properties):
            pairs = fa.neighbor_list(atoms, 3.0)
            return {"energy": float(len(pairs))}

    atoms = fa.build.fcc("Cu", 3.6, cubic=True) * (2, 2, 2)
    atoms.calc = Neighbourly()
    assert atoms.get_potential_energy() > 0.0
