"""The Atoms container: properties, arrays, info and ASE conversion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fastatoms


def _water_box() -> Any:
    cell = np.diag([3.0, 4.0, 5.0])
    positions = np.array([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5], [3.0, 3.5, 4.0]])
    return fastatoms.Atoms(numbers=["H", "O", "H"], positions=positions, cell=cell, pbc=True)


def test_basic_properties() -> None:
    atoms = _water_box()
    assert len(atoms) == 3
    assert atoms.numbers.tolist() == [1, 8, 1]
    assert atoms.positions.shape == (3, 3)
    assert atoms.cell.shape == (3, 3)
    assert atoms.pbc.tolist() == [True, True, True]
    assert atoms.symbols == ["H", "O", "H"]
    assert atoms.labels == ["H", "O", "H"]
    assert atoms.formula == "H2O"
    assert atoms.volume == pytest.approx(60.0)


def test_positions_are_copies() -> None:
    atoms = _water_box()
    view = atoms.positions
    view[0, 0] = 99.0
    assert atoms.positions[0, 0] == 0.0


def test_scaled_positions_wrap() -> None:
    atoms = _water_box()
    atoms.positions = atoms.positions + np.array([3.0, 0.0, 0.0])
    scaled = atoms.scaled_positions(wrap=True)
    assert np.all(scaled >= -1e-9) and np.all(scaled < 1.0 + 1e-9)
    np.testing.assert_allclose(scaled[1], [0.5, 0.5, 0.5], atol=1e-12)


def test_repeat_tuple_and_int() -> None:
    atoms = _water_box()
    supercell = atoms.repeat((2, 2, 2))
    assert len(supercell) == 8 * len(atoms)
    np.testing.assert_allclose(np.diag(np.asarray(supercell.cell)), [6.0, 8.0, 10.0])
    assert len(atoms.repeat(1)) == len(atoms)


def test_arrays_strings_are_lists_and_info_is_a_dict() -> None:
    atoms = _water_box()
    atoms.set_array("forces", np.ones((3, 3)))
    atoms.set_array("tag_int", np.array([1, 2, 3]))
    atoms.set_array("names", ["a", "b", "c"])
    atoms.info = {"energy": -1.5, "comment": "test", "nested": {"k": [1, 2.0, None]}}

    arrays = atoms.arrays
    assert arrays["forces"].shape == (3, 3)
    assert arrays["tag_int"].dtype.kind == "i"
    assert arrays["names"] == ["a", "b", "c"]
    assert atoms.info == {"energy": -1.5, "comment": "test", "nested": {"k": [1, 2.0, None]}}


def test_builtin_arrays_round_trip() -> None:
    atoms = _water_box()
    atoms.masses = np.array([1.0, 16.0, 1.0])
    atoms.tags = np.array([0, 1, 0])
    atoms.charges = np.array([0.1, -0.2, 0.1])
    arrays = atoms.arrays
    assert {"masses", "tags", "charges"} <= set(arrays)
    np.testing.assert_allclose(arrays["masses"], [1.0, 16.0, 1.0])


def test_bad_shapes_are_value_errors() -> None:
    with pytest.raises(ValueError):
        fastatoms.Atoms(numbers=[1, 2], positions=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        fastatoms.Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]], cell=np.zeros((2, 2)))


def test_to_ase_keeps_labels_and_arrays() -> None:
    pytest.importorskip("ase")
    atoms = _water_box()
    atoms.set_array("labels", ["O1", "H1", "H2"])
    atoms.set_array("forces", np.zeros((len(atoms), 3)))
    ase_atoms = atoms.to_ase()
    assert list(ase_atoms.arrays["labels"]) == ["O1", "H1", "H2"]
    assert ase_atoms.arrays["forces"].shape == (len(atoms), 3)
    np.testing.assert_allclose(ase_atoms.positions, atoms.positions)
    np.testing.assert_allclose(ase_atoms.cell.array, np.asarray(atoms.cell))


def test_from_ase_round_trip() -> None:
    pytest.importorskip("ase")
    import ase

    original = ase.Atoms(
        "H2O",
        positions=[[0, 0, 0], [0.9, 0.1, 0], [-0.3, 0.9, 0]],
        cell=[4, 5, 6],
        pbc=[True, False, True],
    )
    original.arrays["momenta"] = np.arange(9.0).reshape(3, 3)
    original.arrays["labels"] = np.array(["O1", "H", "H2"])
    original.info["energy"] = np.float64(-14.2)

    atoms = fastatoms.Atoms.from_ase(original)
    assert atoms.numbers.tolist() == original.numbers.tolist()
    np.testing.assert_allclose(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.pbc, original.pbc)
    assert atoms.labels == ["O1", "H", "H2"]
    np.testing.assert_allclose(atoms.momenta, original.arrays["momenta"])
    assert atoms.info["energy"] == pytest.approx(-14.2)

    back = atoms.to_ase()
    np.testing.assert_allclose(back.positions, original.positions)
    assert list(back.arrays["labels"]) == ["O1", "H", "H2"]
