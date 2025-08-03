#include <torch/extension.h>
#include "cpu/ggnn_cpu.h" // Our clean C++ API header

// PYBIND11_MODULE is a macro that creates an entry point for the Python interpreter to load.
// The name TORCH_EXTENSION_NAME is a special variable that will be replaced by
// the `name` you specified in your setup.py (i.e., 'ggnn_torch_extension._C').
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Set a docstring for the module, which is helpful for introspection.
    m.doc() = "GGNN (Graph-based GPU Nearest Neighbor Search) C++/CUDA backend";

    // Bind the DistanceMeasure enum so we can use it from Python.
    py::enum_<ggnn::DistanceMeasure>(m, "DistanceMeasure")
        .value("Euclidean", ggnn::DistanceMeasure::Euclidean)
        .value("Cosine", ggnn::DistanceMeasure::Cosine)
        .export_values(); // Makes the enum values available in the module's namespace

    // Bind the 'build_graph_cpp' function to the Python name 'build_graph'.
    m.def(
        "build_graph", &ggnn::build_graph_cpp,
        "Builds the GGNN search graph on the GPU.",
        // Use py::arg() to name the arguments for better Python ergonomics.
        py::arg("base_tensor"),
        py::arg("k_build"),
        py::arg("tau_build"),
        py::arg("refinement_iterations"),
        py::arg("measure")
    );

    // Bind the 'query_graph_cpp' function to the Python name 'query_graph'.
    m.def(
        "query_graph", &ggnn::query_graph_cpp,
        "Queries the GGNN search graph on the GPU.",
        py::arg("query_tensor"),
        py::arg("graph_tensor"),
        py::arg("base_tensor"),
        py::arg("k_build"),
        py::arg("k_query"),
        py::arg("tau_query"),
        py::arg("max_iterations"),
        py::arg("measure")
    );
    
    // Bind the 'bf_query_cpp' function to the Python name 'bf_query'.
    m.def(
        "bf_query", &ggnn::bf_query_cpp,
        "Performs a brute-force (exact) k-NN search on the GPU.",
        py::arg("base_tensor"),
        py::arg("query_tensor"),
        py::arg("k"),
        py::arg("measure")
    );
}