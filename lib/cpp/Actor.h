#pragma once

#include "ActorWorker.h"
#include <memory>


class Actor {
public:
    Actor(
            const open_spiel::Game * game_,
            torch::jit::Module & model_,
            size_t num_workers_,
            size_t num_worker_threads_,
            size_t batch_size_,
            size_t max_queue_capacity
    )
        : game(game_)
        , model(model_)
        , num_workers(num_workers_)
        , num_worker_threads(num_worker_threads_)
        , batch_size(batch_size_)
        , queue(std::make_unique<ActorWorker::Queue>(max_queue_capacity))
    {
    };

    void run() {
        for (size_t i = 0; i < num_workers; ++i) {
            workers.push_back(std::make_unique<ActorWorker>(
                game, model, queue.get(), batch_size, num_worker_threads
            ));
            auto thread = std::thread(&ActorWorker::run, workers[i].get());
            threads.emplace_back(std::move(thread));
        }
    }

    void stop() {
        for (size_t i = 0; i < num_workers; ++i) workers[i]->stop();
        for (size_t i = 0; i < num_workers; ++i) threads[i].join();
    }

private:
    const open_spiel::Game * game;
    torch::jit::Module model;
    const size_t num_workers;
    const size_t num_worker_threads;
    const size_t batch_size;
    std::unique_ptr<ActorWorker::Queue> queue;

    std::vector< std::unique_ptr<ActorWorker> > workers;
    std::vector<std::thread> threads;
};