#pragma once

#include <torch/script.h>

#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/games/universal_poker/universal_poker.h"

using namespace open_spiel::universal_poker;


class RNadBot : public open_spiel::Bot {
public:
    RNadBot(int seed, torch::jit::Module * model_);
    ~RNadBot() = default;

    void RestartAt(const open_spiel::State&) override {}
    open_spiel::Action Step(const open_spiel::State& state) override {
        return StepWithPolicy(state).second;
    }

    bool ProvidesPolicy() override { return true; }
    open_spiel::ActionsAndProbs GetPolicy(const open_spiel::State& state) override;

    std::pair<open_spiel::ActionsAndProbs, open_spiel::Action> StepWithPolicy(
        const open_spiel::State& state) override {
        open_spiel::ActionsAndProbs actions_and_probs = GetPolicy(state);
        return {actions_and_probs, open_spiel::SampleAction(actions_and_probs, rng).first};
    }

    bool IsClonable() const override { return true; }
    std::unique_ptr<Bot> Clone() override {
        return std::make_unique<RNadBot>(*this);
    }
    RNadBot(const RNadBot& other) = default;

private:
    mutable torch::jit::script::Module * model;
    std::mt19937 rng;
};
