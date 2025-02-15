#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

#include "Actor.h"
#include "open_spiel/spiel.h"

namespace py = pybind11;


PYBIND11_MODULE(poker_rnad_py, m) {
    py::class_<Trajectory::State>(m, "State")
        .def_readwrite("information_state", &Trajectory::State::information_state)
        .def_readwrite("current_player", &Trajectory::State::current_player)
        .def_readwrite("legal_actions", &Trajectory::State::legal_actions)
        .def_readwrite("is_terminal", &Trajectory::State::is_terminal)
        .def_readwrite("policy", &Trajectory::State::policy)
        .def_readwrite("action", &Trajectory::State::action)
        .def_readwrite("returns", &Trajectory::State::returns)

        .def("__repr__", &Trajectory::State::ToString)
        .def("__str__", &Trajectory::State::ToString);

    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<>())
        .def_readwrite("states", &Trajectory::states)
        .def("__repr__", &Trajectory::ToString)
        .def("__str__", &Trajectory::ToString);

    py::class_<TrajectoryTensors>(m, "TrajectoryTensors")
        .def_readwrite("information_state", &TrajectoryTensors::information_state)
        .def_readwrite("current_player", &TrajectoryTensors::current_player)
        .def_readwrite("legal_actions", &TrajectoryTensors::legal_actions)
        .def_readwrite("is_terminal", &TrajectoryTensors::is_terminal)
        .def_readwrite("policy", &TrajectoryTensors::policy)
        .def_readwrite("action", &TrajectoryTensors::action)
        .def_readwrite("returns", &TrajectoryTensors::returns);

    py::class_<Actor>(m, "Actor")
        .def(py::init<const std::string &, const torch::jit::Module &, const size_t,
            const size_t, const size_t, const size_t, const std::string_view>(),
            py::arg("game"), py::arg("model"), py::arg("num_workers"),
            py::arg("num_threads"), py::arg("batch_size"),
            py::arg("max_queue_capacity"), py::arg("device_name") = "cpu")
        .def("run", &Actor::run)
        .def("stop", &Actor::stop)
        .def("get_batch", &Actor::getBatch, py::arg("wait_seconds"))
        .def("update_model", &Actor::updateModel, py::arg("model"));
}