#include "snn.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <random>

bool SNNMemento::IsValid() const {
    if (i_n_ <= 0) return false;
    if (h_l_ <= 0) return false;
    if (h_n_ <= 0) return false;
    if (o_n_ <= 0) return false;
    if (input_layer.size() != i_n_) return false;
    if (hidden_layers.size() != h_l_) return false;
    for (const auto& layer : hidden_layers) {
        if (layer.size() != h_n_) return false;
    }
    if (output_layer.size() != o_n_) return false;
    if (weights.size() != h_l_ + 1) return false;
    for (size_t l = 0; l < h_l_ + 1; ++l) {
        if (l == 0) {
            if (weights[l].size() != h_n_) return false;
            for (const auto& edges : weights[l]) {
                if (edges.size() != i_n_) return false;
            }
        } else if (l == h_l_) {
            if (weights[l].size() != o_n_) return false;
            for (const auto& edges : weights[l]) {
                if (edges.size() != h_n_) return false;
            }
        } else {
            if (weights[l].size() != h_n_) return false;
            for (const auto& edges : weights[l]) {
                if (edges.size() != h_n_) return false;
            }
        }
    }
    if (biases.size() != h_l_ + 1) return false;
    for (size_t l = 0; l <= h_l_; ++l) {
        size_t layer_size = (l == h_l_) ? o_n_ : h_n_;
        if (biases[l].size() != layer_size) return false;
    }
    if (errors.size() != h_l_ + 1) return false;
    for (size_t l = 0; l <= h_l_; ++l) {
        size_t layer_size = (l == h_l_) ? o_n_ : h_n_;
        if (errors[l].size() != layer_size) return false;
    }
    return true;
}

SNN::SNN(size_t i_n, size_t h_l, size_t h_n, size_t o_n)
: i_n_(i_n)
, h_l_(h_l)
, h_n_(h_n)
, o_n_(o_n) {
    assert(i_n > 0);
    assert(h_l > 0);
    assert(h_n > 0);
    assert(o_n > 0);

    // Initialize the input layer
    input_layer.resize(i_n);

    // Initialize the hidden layers
    hidden_layers.resize(h_l);
    for (auto& layer : hidden_layers) {
        layer.resize(h_n);
    }

    // Initialize the output layer
    output_layer.resize(o_n);

    // Initialize the weights
    weights.resize(h_l + 1);
    for (size_t l = 0; l < h_l + 1; ++l) {
        if (l == 0) {
            weights[l].resize(h_n);
            for (auto& edges : weights[l]) {
                edges.resize(i_n);
            }
        } else if (l == h_l) {
            weights[l].resize(o_n);
            for (auto& edges : weights[l]) {
                edges.resize(h_n);
            }
        } else {
            weights[l].resize(h_n);
            for (auto& edges : weights[l]) {
                edges.resize(h_n);
            }
        }
    }

    // Initialize biases
    biases.resize(h_l + 1);
    for (size_t l = 0; l <= h_l; ++l) {
        size_t layer_size = (l == h_l) ? o_n : h_n;
        biases[l].resize(layer_size);
    }

    // Initialize errors
    errors.resize(h_l + 1);
    for (size_t l = 0; l <= h_l; ++l) {
        size_t layer_size = (l == h_l) ? o_n : h_n;
        errors[l].resize(layer_size);
    }
}

SNN::SNN(const SNNMemento& memento) {
    RestoreFromMemento(memento);
}

SNNMemento SNN::CreateMemento() const {
    SNNMemento memento;
    memento.input_layer = input_layer;
    memento.hidden_layers = hidden_layers;
    memento.output_layer = output_layer;
    memento.weights = weights;
    memento.biases = biases;
    memento.errors = errors;
    memento.i_n_ = i_n_;
    memento.h_l_ = h_l_;
    memento.h_n_ = h_n_;
    memento.o_n_ = o_n_;
    memento.eta_ = eta_;
    return memento;
}

void SNN::RestoreFromMemento(const SNNMemento& memento) {
    using namespace std::literals;
    if (!memento.IsValid()) {
        throw std::runtime_error("It is impossible to restore the internal state. "
                                 "The data format is not correct"s);
    }

    input_layer = memento.input_layer;
    hidden_layers = memento.hidden_layers;
    output_layer = memento.output_layer;
    weights = memento.weights;
    biases = memento.biases;
    errors = memento.errors;
    i_n_ = memento.i_n_;
    h_l_ = memento.h_l_;
    h_n_ = memento.h_n_;
    o_n_ = memento.o_n_;
    eta_ = memento.eta_;
}

void SNN::InitializeWeightsWithRandom() {
    std::random_device rd;
    std::default_random_engine e2(rd());
    std::normal_distribution<float> dist(0, std::sqrt(2.0f / h_n_));
    for (auto& layer : weights) {
        for (auto& row : layer) {
            for (auto& weight : row) {
                weight = dist(e2);
            }
        }
    }
}

void SNN::InitializeBiasesWithRandom(float min, float max) {
    std::random_device rd;
    std::default_random_engine e2(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for (auto& layer : biases) {
        for (auto& bias : layer) {
            bias = dist(e2);
        }
    }
}

void SNN::CalculateOutput(const std::vector<float>& input) noexcept {
    assert(input.size() == i_n_);
    input_layer = input;

    // Calculate hidden layers outputs
    for (size_t l = 0; l < h_l_; ++l) {
        for (size_t i = 0; i < h_n_; ++i) {
            float net_i = 0;
            const auto& prev_layer = (l == 0) ? input_layer : hidden_layers[l - 1];
            for (size_t j = 0; j < prev_layer.size(); ++j) {
                net_i += weights[l][i][j] * prev_layer[j];
            }
            net_i += biases[l][i];
            hidden_layers[l][i] = 1.0f / (1.0f + std::exp(-net_i)); // sigmoid activation
        }
    }

    // Calculate output layer outputs
    for (size_t i = 0; i < o_n_; ++i) {
        float net_i = 0;
        for (size_t j = 0; j < h_n_; ++j) {
            net_i += weights[h_l_][i][j] * hidden_layers.back()[j];
        }
        net_i += biases.back()[i];
        output_layer[i] = 1.0f / (1.0f + std::exp(-net_i)); // sigmoid activation
    }
}

float SNN::EvaluateError(const std::vector<float>& target) const {
    assert(target.size() == o_n_);

    float result = 0.0f;
    // Use RMSE (root mean squared error) to calculate
    for (size_t i = 0; i < o_n_; ++i) {
        float delta = target[i] - output_layer[i];
        result += delta * delta;
    }
    return std::sqrt(result / o_n_);
}

void SNN::PropagateErrorBack(const std::vector<float>& target) noexcept {
    assert(target.size() == o_n_);

    // Calculate the error on the output layer
    for (size_t i = 0; i < o_n_; ++i) {
        float out = output_layer[i];
        float delta = target[i] - out;
        errors.back()[i] = out * (1 - out) * delta;
    }

    // Calculate errors for hidden layers
    for (int l = h_l_ - 1; l >= 0; --l) {
        std::fill(errors[l].begin(), errors[l].end(), 0.0f);

        for (size_t j = 0; j < errors[l + 1].size(); ++j) {
            float error_next = errors[l + 1][j];
            for (size_t i = 0; i < h_n_; ++i) {
                errors[l][i] += weights[l + 1][j][i] * error_next;
            }
        }

        for (size_t i = 0; i < h_n_; ++i) {
            float out = hidden_layers[l][i];
            errors[l][i] *= out * (1 - out);
        }
    }

    // Update weights
    for (size_t l = 0; l < h_l_ + 1; ++l) {
        const auto& prev_layer = (l == 0) ? input_layer : hidden_layers[l - 1];
        for (size_t i = 0; i < errors[l].size(); ++i) {
            float error = errors[l][i];
            for (size_t j = 0; j < prev_layer.size(); ++j) {
                weights[l][i][j] += eta_ * error * prev_layer[j];
            }
            biases[l][i] += eta_ * error;
        }
    }
}

const std::vector<float>& SNN::ReadOutput() const {
    return output_layer;
}

void SNN::SetLearningCoefficient(float eta) {
    eta_ = eta;
}

namespace tests {

void Propagate() {
    auto set_unit = [](std::vector<float>& vec, size_t pos) {
        std::fill(vec.begin(), vec.end(), 0.0f);
        vec[pos] = 1;
    };

    SNN snn(10, 1, 10, 10);
    snn.InitializeWeightsWithRandom();
    snn.InitializeBiasesWithRandom();
    snn.SetLearningCoefficient(1.0f);

    // Propagate
    std::vector<float> resources(10);
    for (size_t t = 0; t < 3000; ++t) {
        set_unit(resources, t % 10);
        snn.CalculateOutput(resources);
        set_unit(resources, 9 - t % 10); // flip vector
        snn.PropagateErrorBack(resources);
    }

    // Check that the network has learned to flip vector
    for (size_t i = 0; i < 10; ++i) {
        set_unit(resources, i);
        snn.CalculateOutput(resources);
        const std::vector<float>& output = snn.ReadOutput();
        auto it = std::max_element(output.begin(), output.end());
        size_t index = it - output.begin();
        assert(index == 9 - i);
    }
}
}