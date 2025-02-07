#pragma once

#include "ActorWorker.h"
#include <memory>


class Actor {
public:
    Actor(
            const open_spiel::Game * game_,
            const torch::jit::Module & model_,
            size_t num_workers_,
            size_t num_worker_threads_,
            size_t batch_size_,
            size_t max_queue_capacity_,
            const std::string_view device_name_ = "cpu"
    )
        : game(game_)
        , model(model_)
        , num_workers(num_workers_)
        , num_worker_threads(num_worker_threads_)
        , batch_size(batch_size_)
        , max_queue_capacity(max_queue_capacity_)
        , device_name(device_name_)
        , running(false)
    {
    };

    ~Actor() { stop(); }

    void run() {
        queue = std::make_unique<ActorWorker::Queue>(max_queue_capacity);

        for (size_t i = 0; i < num_workers; ++i) {
            workers.push_back(std::make_unique<ActorWorker>(
                game,
                model,
                queue.get(),
                batch_size,
                num_worker_threads,
                device_name
            ));
            auto thread = std::thread(&ActorWorker::run, workers[i].get());
            threads.emplace_back(std::move(thread));
        }

        running = true;
    }

    void stop() {
        for (size_t i = 0; i < workers.size(); ++i) workers[i]->stop();
        for (size_t i = 0; i < threads.size(); ++i) threads[i].join();

        threads.clear();
        workers.clear();

        running = false;
    }

    ActorWorker::TrajectoryBatch getBatch(size_t seconds) {
        if (!running) throw std::runtime_error("Actor not running");
        auto batch = queue->Pop(absl::Seconds(seconds));
        if (!batch.has_value()) throw std::runtime_error("Timeout waiting for batch");
        return batch.value();
    }

    void updateModel(const torch::jit::Module & model) {
        for (size_t i = 0; i < workers.size(); ++i) workers[i]->updateModel(model);
    }

private:
    const open_spiel::Game * game;
    torch::jit::Module model;
    const size_t num_workers;
    const size_t num_worker_threads;
    const size_t batch_size;
    const size_t max_queue_capacity;
    const std::string device_name;
    std::unique_ptr<ActorWorker::Queue> queue;
    bool running = false;

    std::vector< std::unique_ptr<ActorWorker> > workers;
    std::vector<std::thread> threads;
};
