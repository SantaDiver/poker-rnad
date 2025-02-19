#include <torch/types.h>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

struct Trajectory {
    struct State {
        std::vector<float> information_state;
        open_spiel::Player current_player;
        std::vector<int> legal_actions;
        bool is_terminal;
        std::vector<double> policy;
        open_spiel::Action action;
        std::vector<double> returns;

        std::string ToString() const {
            std::string s;
            absl::StrAppend(&s, "State(current_player=", current_player,
                ", action=", action, ", is_terminal=", is_terminal, ")");
            return s;
        }
    };

    std::vector<State> states;

    std::string ToString() const {
        std::string s;
        absl::StrAppend(&s, "Trajectory(", states.size(), " states)");
        return s;
    }
};

struct TrajectoryTensors {
    // T - num timeframes, shorter sequences are padded
    // B - batch size
    torch::Tensor information_state;  // [T, B, I] I - infromation state size
    torch::Tensor current_player;  // [T, B]
    torch::Tensor legal_actions;  // [T, B, A] A - number of distinct actions
    torch::Tensor is_terminal;  // [T, B]
    torch::Tensor policy;  // [T, B, A] A - number of distinct actions
    torch::Tensor action;  // [T, B]
    torch::Tensor returns;  // [T, B, P] P - number of players

    TrajectoryTensors(int64_t T, int64_t B, int64_t I, int64_t A, int64_t P)
        : information_state(torch::zeros({T, B, I}, torch::kFloat32))
        , current_player(torch::zeros({T, B}, torch::kInt64))
        , legal_actions(torch::zeros({T, B, A}, torch::kBool))
        , is_terminal(torch::zeros({T, B}, torch::kInt64))
        , policy(torch::zeros({T, B, A}, torch::kFloat32))
        , action(torch::zeros({T, B}, torch::kInt64))
        , returns(torch::zeros({T, B, P}, torch::kFloat32))
    {
    }
};
