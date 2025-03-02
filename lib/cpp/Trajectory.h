#include <torch/types.h>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"

using open_spiel::universal_poker::UniversalPokerState;
using open_spiel::universal_poker::StateActionType;
using open_spiel::universal_poker::acpc_cpp::ACPCState;


struct InformationState {
    static constexpr uint16_t BET_TO_POT_MAX_BUCKET = 9;
    static constexpr uint16_t STACK_TO_POT_MAX_BUCKET = 9;

    open_spiel::Player current_player;
    std::vector<uint16_t> stack_to_pot_ratios;
    uint16_t amount_to_call_to_stack_ratio;

    StateActionType action_type;
    int size;

    InformationState(const open_spiel::State * state, open_spiel::Action action)
        : current_player(state->CurrentPlayer())
    {
        const UniversalPokerState * poker_state = dynamic_cast<const UniversalPokerState *>(state);
        const ACPCState & acpc_state = poker_state->acpc_state();

        for (open_spiel::Player player = 0; player < state->NumPlayers(); ++player) {
            const uint16_t stack = acpc_state.Money(player);
            const uint16_t pot = acpc_state.TotalSpent();
            stack_to_pot_ratios[player] = stack * STACK_TO_POT_MAX_BUCKET / pot;
            stack_to_pot_ratios[player] = std::min(stack_to_pot_ratios[player], STACK_TO_POT_MAX_BUCKET);
        }

        const auto to_call = acpc_state.MaxSpend() - acpc_state.CurrentSpent(current_player);
        const uint16_t to_call_bucket = to_call * BET_TO_POT_MAX_BUCKET / acpc_state.Money(current_player);
        amount_to_call_to_stack_ratio = std::min(to_call_bucket, BET_TO_POT_MAX_BUCKET);

        int action_int = static_cast<int>(action);
        if (action_int == open_spiel::universal_poker::kFold) {
            action_type = open_spiel::universal_poker::ACTION_FOLD;
            size = 0;
        } else if (action_int == open_spiel::universal_poker::kCall) {
            action_type = open_spiel::universal_poker::ACTION_CHECK_CALL;
            size = 0;
        } else {
            action_type = open_spiel::universal_poker::ACTION_BET;
            size = action_int;
        }
    }
};

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
