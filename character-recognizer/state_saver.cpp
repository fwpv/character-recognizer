#include "state_saver.h"

#include <fstream>
#include <stdexcept>
#include <vector>

using namespace std::literals;

template <typename T>
void SaveVector(std::ofstream& out, const std::vector<T>& vec) {
    size_t size = vec.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
}

template <typename T>
void LoadVector(std::ifstream& in, std::vector<T>& vec) {
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
}

constexpr uint32_t current_version = 0x24052823;

void SaveSnnState(const std::filesystem::path& file, const SnnMemento& state) {
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to open file "s + file.string() + " for saving state"s);
    }

    out.write(reinterpret_cast<const char*>(&current_version), sizeof(current_version));

    // Write snn memento
    out.write(reinterpret_cast<const char*>(&state.i_n), sizeof(state.i_n));
    out.write(reinterpret_cast<const char*>(&state.h_l), sizeof(state.h_l));
    out.write(reinterpret_cast<const char*>(&state.h_n), sizeof(state.h_n));
    out.write(reinterpret_cast<const char*>(&state.o_n), sizeof(state.o_n));
    out.write(reinterpret_cast<const char*>(&state.eta), sizeof(state.eta));

    for (const auto& layer : state.layers) {
        SaveVector(out, layer);
    }

    for (const auto& layer : state.weights) {
        for (const auto& edges : layer) {
            SaveVector(out, edges);
        }
    }

    for (const auto& vec : state.biases) {
        SaveVector(out, vec);
    }

    for (const auto& vec : state.errors) {
        SaveVector(out, vec);
    }
}

SnnMemento LoadSnnState(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to open file "s + file.string() + " for loading state"s);
    }

    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != current_version) {
        throw std::runtime_error("Version of file "s + file.string() + " is not supported"s);
    }

    // Read snn memento
    SnnMemento state;

    in.read(reinterpret_cast<char*>(&state.i_n), sizeof(state.i_n));
    in.read(reinterpret_cast<char*>(&state.h_l), sizeof(state.h_l));
    in.read(reinterpret_cast<char*>(&state.h_n), sizeof(state.h_n));
    in.read(reinterpret_cast<char*>(&state.o_n), sizeof(state.o_n));
    in.read(reinterpret_cast<char*>(&state.eta), sizeof(state.eta));

    state.layers.resize(state.h_l + 2);
    for (auto& layer : state.layers) {
        LoadVector(in, layer);
    }

    state.weights.resize(state.h_l + 1);
    for (size_t l = 0; l < state.h_l + 1; ++l) {
        auto& layer = state.weights[l];
        if (l == state.h_l) {
            layer.resize(state.o_n);
        } else {
            layer.resize(state.h_n);
        }
        for (auto& edges : layer) {
            LoadVector(in, edges);
        }
    }

    state.biases.resize(state.h_l + 1);
    for (auto& vec : state.biases) {
        LoadVector(in, vec);
    }

    state.errors.resize(state.h_l + 1);
    for (auto& vec : state.errors) {
        LoadVector(in, vec);
    }

    return state;
}