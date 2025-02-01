#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Actor.h"

namespace py = pybind11;


PYBIND11_MODULE(poker_rnad_py, m) {
    py::class_<Trajectory::State>(m, "State")
        .def_readwrite("information_state", &Trajectory::State::information_state)
        .def_readwrite("current_player", &Trajectory::State::current_player)
        .def_readwrite("legal_actions", &Trajectory::State::legal_actions)
        .def_readwrite("action", &Trajectory::State::action)
        .def_readwrite("policy", &Trajectory::State::policy)

        .def("__repr__", &Trajectory::State::ToString)
        .def("__str__", &Trajectory::State::ToString);

    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<>())
        .def_readwrite("states", &Trajectory::states)
        .def_readwrite("returns", &Trajectory::returns)
        .def("__repr__", &Trajectory::ToString)
        .def("__str__", &Trajectory::ToString);

    py::class_<Actor>(m, "Actor")
        .def(py::init<const open_spiel::Game &, torch::jit::Module &>())
        .def("generate_trajectory", &Actor::generateTrajectory);
}
