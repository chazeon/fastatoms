// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include <memory>
#include <string>

#include "fastatoms/calculators/mlip.hpp"
#include "fastatoms/calculators/mlip_models.hpp"
#include "fastatoms/calculators/mlip_torch.hpp"
#include "fastatoms/calculators/mlip_worker.hpp"

namespace fastatoms::bindings {
namespace {

using namespace fastatoms::mlip;

/// A backend written in Python: anything with an `evaluate(graph, need_forces,
/// need_stress)` method and an `info` attribute.
class PythonBackend final : public Backend {
public:
    PythonBackend(py::object implementation, ModelInfo info)
        : implementation_(std::move(implementation)), info_(std::move(info)) {}

    [[nodiscard]] const ModelInfo& info() const override { return info_; }

    [[nodiscard]] bool needs_edges() const override {
        return !py::hasattr(implementation_, "needs_edges") ||
               implementation_.attr("needs_edges").cast<bool>();
    }

    [[nodiscard]] Prediction evaluate(const Graph& graph, bool need_forces,
                                      bool need_stress) override {
        py::gil_scoped_acquire gil;
        const py::object answer =
            implementation_.attr("evaluate")(graph, need_forces, need_stress);

        Prediction prediction;
        if (py::isinstance<py::dict>(answer)) {
            const auto mapping = answer.cast<py::dict>();
            prediction.energy = mapping["energy"].cast<double>();
            if (mapping.contains("forces") && !mapping["forces"].is_none()) {
                prediction.forces = as_flat_doubles(
                    py::reinterpret_borrow<py::object>(mapping["forces"]),
                    static_cast<py::ssize_t>(graph.natoms), 3, "forces");
                prediction.has_forces = true;
            }
            if (mapping.contains("stress") && !mapping["stress"].is_none()) {
                const std::vector<double> voigt = as_flat_doubles(
                    py::reinterpret_borrow<py::object>(mapping["stress"]), 6, 1, "stress");
                std::copy(voigt.begin(), voigt.end(), prediction.stress.begin());
                prediction.has_stress = true;
            }
            return prediction;
        }
        throw MLIPError("mlip: a Python backend must return a dict with an 'energy' key");
    }

private:
    py::object implementation_;
    ModelInfo info_;
};

/// Keyword arguments become the parameter mapping, exactly as for the DFT
/// calculators, so a model reads the same from Python and from YAML.
Value from_kwargs(const py::kwargs& kwargs) { return py::cast<Value>(py::dict(kwargs)); }

}  // namespace

void bind_mlip(py::module_& m) {
    // ---------------------------------------------------------------------
    // Model metadata
    // ---------------------------------------------------------------------
    py::class_<ModelInfo>(m, "MLIPModelInfo", "What a backend reports about its loaded model.")
        .def_readonly("family", &ModelInfo::family)
        .def_readonly("path", &ModelInfo::path)
        .def_readonly("backend", &ModelInfo::backend,
                      "'torchscript' for in-process LibTorch, 'worker' for a Python process.")
        .def_readonly("device", &ModelInfo::device)
        .def_readonly("dtype", &ModelInfo::dtype)
        .def_readonly("version", &ModelInfo::version)
        .def_readonly("cutoff", &ModelInfo::cutoff, "Interaction radius in angstrom.")
        .def_readonly("species", &ModelInfo::species,
                      "Atomic numbers the model was trained on; empty means unrestricted.")
        .def_readonly("provides_forces", &ModelInfo::provides_forces,
                      "False means FastAtoms differentiates the energy for them.")
        .def_readonly("provides_stress", &ModelInfo::provides_stress)
        .def("supports", &ModelInfo::supports, py::arg("atomic_number"))
        .def("__repr__", &ModelInfo::to_string);

    // ---------------------------------------------------------------------
    // The graph
    // ---------------------------------------------------------------------
    py::class_<Graph>(m, "MLIPGraph", R"doc(
An ``Atoms`` object in the form a message-passing potential consumes.

    graph = fa.mlip_graph(atoms, cutoff=5.0)
    graph.edge_index.shape        # (2, nedges): receivers, then senders
    graph.unit_shifts.shape       # (nedges, 3), integer cell offsets

The convention is the one MACE, MatterSim, CHGNet and NequIP share::

    r_ij = positions[sender] + unit_shifts @ cell - positions[receiver]
)doc")
        .def_readonly("natoms", &Graph::natoms)
        .def_readonly("cell", &Graph::cell)
        .def_readonly("cutoff", &Graph::cutoff)
        .def_property_readonly("nedges", &Graph::nedges)
        .def_property_readonly(
            "positions",
            [](const Graph& self) {
                std::vector<double> values = self.positions;
                const auto n = static_cast<py::ssize_t>(self.natoms);
                return move_to_array(std::move(values), {n, static_cast<py::ssize_t>(3)});
            },
            "(natoms, 3) in angstrom.")
        .def_property_readonly(
            "atomic_numbers",
            [](const Graph& self) {
                std::vector<std::int64_t> values = self.numbers;
                const auto n = static_cast<py::ssize_t>(self.natoms);
                return move_to_array(std::move(values), {n});
            })
        .def_property_readonly("pbc",
                               [](const Graph& self) { return pbc_to_array(self.pbc); })
        .def_property_readonly(
            "edge_index",
            [](const Graph& self) {
                std::vector<std::int64_t> values = self.edge_index;
                const auto m = static_cast<py::ssize_t>(self.nedges());
                return move_to_array(std::move(values), {static_cast<py::ssize_t>(2), m});
            },
            "(2, nedges): receivers on row 0, senders on row 1.")
        .def_property_readonly(
            "unit_shifts",
            [](const Graph& self) {
                std::vector<double> values = self.unit_shifts;
                const auto m = static_cast<py::ssize_t>(self.nedges());
                return move_to_array(std::move(values), {m, static_cast<py::ssize_t>(3)});
            },
            "(nedges, 3) integer cell offsets applied to the sender.")
        .def("__len__", [](const Graph& self) { return self.natoms; })
        .def("__repr__", [](const Graph& self) {
            return "MLIPGraph(" + std::to_string(self.natoms) + " atoms, " +
                   std::to_string(self.nedges()) + " edges, cutoff " +
                   std::to_string(self.cutoff) + " A)";
        });

    m.def(
        "mlip_graph",
        [](const Atoms& atoms, double cutoff, bool edges) {
            return build_graph(atoms, cutoff, edges);
        },
        py::arg("atoms"), py::arg("cutoff"), py::arg("edges") = true,
        "Convert a structure into the tensors a neural potential consumes.");

    // ---------------------------------------------------------------------
    // The calculator
    // ---------------------------------------------------------------------
    py::class_<MLIPCalculator, Calculator, std::shared_ptr<MLIPCalculator>> mlip(
        m, "MLIP", R"doc(
A calculator backed by a machine-learning interatomic potential.

Two backends sit behind the same interface. An exported TorchScript archive
runs in this process through LibTorch, with no Python in the loop::

    atoms.calc = fa.MLIP(model="mace-off23-small.pt", family="mace")

Anything else - a checkpoint that has to be unpickled, a model that cannot be
scripted - is driven by one long-lived interpreter in the environment that has
the package installed::

    atoms.calc = fa.MatterSim(model="MatterSim-v1.0.0-1M.pth")

Whatever the model does not return itself is differentiated out of its energy:
forces from the gradient with respect to the positions, stress from the
gradient with respect to a strain applied to the positions and the cell.
)doc");
    mlip.def(py::init([](const py::kwargs& kwargs) {
                 return std::make_shared<MLIPCalculator>(from_kwargs(kwargs));
             }))
        .def("load", &MLIPCalculator::load,
             "Load the model now, so the first step is not charged for it.")
        .def("unload", &MLIPCalculator::unload,
             "Release the model, ending the worker process if there is one.")
        .def_property_readonly("loaded", &MLIPCalculator::loaded)
        .def_property_readonly("model", &MLIPCalculator::model,
                               py::return_value_policy::reference_internal,
                               "Metadata about the loaded model; loads it if necessary.")
        .def_property_readonly("cutoff", &MLIPCalculator::cutoff,
                               "The neighbour-list radius in use, in angstrom.")
        .def_property_readonly("family", &MLIPCalculator::family)
        .def("graph", &MLIPCalculator::graph, py::arg("atoms"),
             "The graph this structure would be sent to the model as.")
        .def(
            "set_backend",
            [](MLIPCalculator& self, const py::object& backend) {
                if (backend.is_none()) {
                    self.set_backend(nullptr);
                    return;
                }
                ModelInfo info;
                info.family = self.family();
                info.backend = "python";
                info.path = "<python>";
                if (py::hasattr(backend, "info")) {
                    const py::object described = backend.attr("info");
                    if (py::isinstance<ModelInfo>(described)) {
                        info = described.cast<ModelInfo>();
                    }
                }
                self.set_backend(std::make_shared<PythonBackend>(backend, std::move(info)));
            },
            py::arg("backend"),
            R"doc(
Evaluate with an object supplied here instead of a model file.

The object needs one method, ``evaluate(graph, need_forces, need_stress)``,
returning a dict with ``energy`` and optionally ``forces`` and ``stress``. An
``info`` attribute holding an :class:`MLIPModelInfo` describes the cutoff and
the species. Passing ``None`` restores the normal behaviour.
)doc");

    // ---------------------------------------------------------------------
    // The families
    // ---------------------------------------------------------------------
    py::class_<MACECalculator, MLIPCalculator, std::shared_ptr<MACECalculator>>(m, "MACE",
                                                                                 R"doc(
MACE: equivariant message passing, and the one family that exports cleanly to
TorchScript - so it is the one that can run entirely in C++.

    atoms.calc = fa.MACE(model="mace-off23-small.pt")   # in process, no Python
    atoms.calc = fa.MACE(model="medium")                # worker, mace_env

Export a checkpoint once to get the fast path::

    mace_env/bin/python -m fastatoms.mlip.export MACE-OFF23_small.model -o mace.pt

A short name like "small" or "medium" is looked up in MACE's own download
cache before being passed on, so a run stays offline when the file is there.
)doc")
        .def(py::init([](const py::kwargs& kwargs) {
            return std::make_shared<MACECalculator>(from_kwargs(kwargs));
        }));

    py::class_<MatterSimCalculator, MLIPCalculator, std::shared_ptr<MatterSimCalculator>>(
        m, "MatterSim", R"doc(
MatterSim, Microsoft's M3GNet-based universal potential.

Runs through the worker: its convolution stack is built with
``functools.partial``, which ``torch.jit.script`` cannot compile, so there is
no archive to hand to LibTorch. One interpreter serves the whole run, so a step
costs a pipe round trip rather than an interpreter start.

    atoms.calc = fa.MatterSim()                              # the 1M checkpoint
    atoms.calc = fa.MatterSim(model="MatterSim-v1.0.0-5M.pth")
)doc")
        .def(py::init([](const py::kwargs& kwargs) {
            return std::make_shared<MatterSimCalculator>(from_kwargs(kwargs));
        }));

    py::class_<CHGNetCalculator, MLIPCalculator, std::shared_ptr<CHGNetCalculator>>(
        m, "CHGNet", R"doc(
CHGNet, the charge-informed universal potential. Driven through the worker.

    atoms.calc = fa.CHGNet()
)doc")
        .def(py::init([](const py::kwargs& kwargs) {
            return std::make_shared<CHGNetCalculator>(from_kwargs(kwargs));
        }));

    py::class_<NequIPCalculator, MLIPCalculator, std::shared_ptr<NequIPCalculator>>(
        m, "NequIP", R"doc(
NequIP, and Allegro which deploys through the same package.

A model compiled with ``nequip-compile`` is TorchScript and runs in process;
anything else goes through the worker.

    atoms.calc = fa.NequIP(model="deployed.pth")
    atoms.calc = fa.NequIP(model="allegro.pth", allegro=True)
)doc")
        .def(py::init([](const py::kwargs& kwargs) {
            Value parameters = from_kwargs(kwargs);
            bool allegro = false;
            if (const Value* value = parameters.find("allegro"); value && value->is_bool()) {
                allegro = value->as_bool();
            }
            return std::make_shared<NequIPCalculator>(std::move(parameters), allegro);
        }));

    m.def("Allegro",
          [](const py::kwargs& kwargs) {
              return std::make_shared<NequIPCalculator>(from_kwargs(kwargs), true);
          },
          "Allegro, deployed through the nequip package.");

    // ---------------------------------------------------------------------
    // Build and environment information
    // ---------------------------------------------------------------------
    m.def("torch_available", &torch_available,
          "True when FastAtoms was built with LibTorch, so TorchScript models can run "
          "in process.");
    m.def("torch_version", &torch_version,
          "The LibTorch version FastAtoms was built against, or an empty string.");
    m.def("mlip_worker_script", &worker_script_path,
          "Path of the Python worker script the out-of-process backend runs.");
    m.def("conda_python", &conda_python, py::arg("environment"),
          "Interpreter for a conda environment name, or '' when it cannot be found.");
    m.def("mlip_environment_for", &default_environment_for, py::arg("family"),
          "The conda environment a family uses by default: mace_env for MACE, "
          "mlip_env for the rest.");
}

}  // namespace fastatoms::bindings
