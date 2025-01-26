#pragma once

#include <torch/script.h>

#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/games/universal_poker/universal_poker.h"

using namespace open_spiel::universal_poker;


class RNadPolicy : public open_spiel::Policy {
public:
    RNadPolicy();
    open_spiel::ActionsAndProbs GetStatePolicy(const open_spiel::State &state) const;

private:
    mutable torch::jit::script::Module model;
};
