#pragma once

#include <memory>
#include <string>
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>

#include "absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

#include "RNadBot.h"
#include "open_spiel/spiel_utils.h"


struct Trajectory {
    struct State {
        std::vector<float> information_state;
        open_spiel::Player current_player;
        std::vector<open_spiel::Action> legal_actions;
        open_spiel::Action action;
        open_spiel::ActionsAndProbs policy;

        std::string ToString() const {
            std::string s;
            absl::StrAppend(&s, "State(current_player=", current_player,
                ", action=", action, ")");
            return s;
        }
    };

    std::vector<State> states;
    std::vector<double> returns;

    std::string ToString() const {
        std::string s;
        absl::StrAppend(&s, "Trajectory(", states.size(), " states)");
        return s;
    }
};


class Actor {
public:
    Actor(const open_spiel::Game & game_, torch::jit::Module & model_)
        : game(game_)
        , model(model_)
        , rng(absl::ToUnixNanos(absl::Now()))
        , bot(std::make_unique<RNadBot>(absl::ToUnixNanos(absl::Now()), &model))
    {
    }

    Trajectory generateTrajectory() const {
        std::unique_ptr<open_spiel::State> state = game.NewInitialState();
        Trajectory trajectory;
        while (!state->IsTerminal()) {
            if (state->IsChanceNode()) {
                open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
                open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
                state->ApplyAction(action);
            } else {
                open_spiel::Player player = state->CurrentPlayer();
                auto action_with_policy = bot->StepWithPolicy(*state.get());
                open_spiel::Action action = action_with_policy.second;

                trajectory.states.push_back(Trajectory::State{
                    state->InformationStateTensor(),
                    player,
                    state->LegalActions(),
                    action_with_policy.second,
                    std::move(action_with_policy.first)
                });
                state->ApplyAction(action);
            }
        }
        trajectory.returns = state->Returns();

        return trajectory;
    };

private:
    const open_spiel::Game & game;
    mutable torch::jit::Module model;
    mutable std::mt19937 rng;
    std::shared_ptr<open_spiel::Policy> policy;
    std::unique_ptr<open_spiel::Bot> bot;
};
