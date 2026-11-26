// SPDX-License-Identifier: MIT
#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"
#include "fastatoms/io/format.hpp"

namespace fastatoms::bindings {

void bind_formats(py::module_& m) {
    // Format keyword options are passed through as a Value map. Unknown keys
    // are rejected by the format itself (FormatError).
    m.def(
        "read",
        [](const std::string& text, const std::string& format, const py::kwargs& kwargs) {
            return io::read(text, format, options_from_kwargs(kwargs));
        },
        py::arg("text"), py::arg("format"),
        "Read one frame from file contents. `index` selects the frame (default -1).");

    m.def(
        "read_all",
        [](const std::string& text, const std::string& format, const py::kwargs& kwargs) {
            return io::read_all(text, format, options_from_kwargs(kwargs));
        },
        py::arg("text"), py::arg("format"), "Every frame in the file contents.");

    m.def(
        "write",
        [](const Atoms& atoms, const std::string& format, const py::kwargs& kwargs) {
            return io::write(atoms, format, options_from_kwargs(kwargs));
        },
        py::arg("atoms"), py::arg("format"), "Serialise one structure to a string.");

    m.def(
        "write_all",
        [](const std::vector<Atoms>& frames, const std::string& format, const py::kwargs& kwargs) {
            return io::write_all(frames, format, options_from_kwargs(kwargs));
        },
        py::arg("frames"), py::arg("format"),
        "Serialise several structures to one text (one block or frame each). Formats that hold "
        "one structure accept a single frame only.");

    m.def("can_write_frames", &io::can_write_frames, py::arg("format"),
          "Whether `format` can hold more than one frame when written.");

    m.def(
        "guess_format",
        [](const std::string& filename, const std::string& head) {
            return io::guess_format(filename, head);
        },
        py::arg("filename"), py::arg("head") = std::string{},
        "Format name from a file name, falling back to the first bytes; '' if unknown.");

    m.def("available_formats", &io::available_formats,
          "Format names compiled into this build.");

    m.def("can_write", &io::can_write, py::arg("format"),
          "Whether `format` has a writer.");
}

}  // namespace fastatoms::bindings
