#include "open_spiel/spiel.h"

struct Trajectory {
    struct State {
        std::vector<float> information_state;
        open_spiel::Player current_player;
        std::vector<uint8_t> legal_actions;
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
