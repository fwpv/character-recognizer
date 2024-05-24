#include "neural_matrix.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

NeuralMatrix::NeuralMatrix(size_t m, size_t n)
: m_(m)
, n_(n) {
    assert(m > 2);
    assert(n > 0);

    // Initialize the neurons
    neurons.resize(m);
    for (auto& layer : neurons) {
        layer.resize(n);
    }

    // Initialize the weights
    weights.resize(m - 1);
    for (auto& layer : weights) {
        layer.resize(n);
        for (auto& edges : layer) {
            edges.resize(n);
        }
    }

    // Initialize vector of errors
    errors.resize(m);
    for (auto& layer : errors) {
        layer.resize(n);
    }

    // Initialize vector of biases
    biases.resize(m - 1);
}

void NeuralMatrix::InitializeWeightsWithRandom() {
    std::random_device rd;
    std::default_random_engine e2(rd());
    std::normal_distribution<float> dist(0, std::sqrt(2.0f / n_));
    for (auto& layer : weights) {
        for (auto& row : layer) {
            for (auto& weight : row) {
                weight = dist(e2);
            }
        }
    }
}

void NeuralMatrix::InitializeBiasesWithRandom(float min, float max) {
    std::random_device rd;
    std::default_random_engine e2(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for (auto& bias : biases) {
        bias = dist(e2);
    }
}

void NeuralMatrix::CalculateOutput(const std::vector<float>& values) {
    assert(values.size() == n_);
    neurons[0] = values;

    for (size_t l = 0; l < m_ - 1; ++l) {
        for (size_t i = 0; i < n_; ++i) {
            float net_i = 0;
            for (size_t j = 0; j < n_; ++j) {
                net_i += weights[l][i][j] * neurons[l][j];
            }
            net_i += biases[l];
            float out_i = 1.0f / (1.0f + std::exp(-net_i)); // sigmoid activation
            neurons[l + 1][i] = out_i;
        }
    }
}

float NeuralMatrix::EvaluateError(const std::vector<float>& target, size_t size) const {
    assert(target.size() <= n_);

    const std::vector<float>& output = neurons.back();
    float result = 0.0f;
    // Use RMSE (root mean squared error) to calculate
    for (size_t i = 0; i < size; ++i) {
        float delta = target[i] - output[i];
        result += delta * delta;
    }
    return std::sqrt(result / n_);
}

float NeuralMatrix::EvaluateError(const std::vector<float>& target) const {
    return EvaluateError(target, n_);
}

void NeuralMatrix::PropagateErrorBack(const std::vector<float>& target) {
    assert(target.size() == n_);

    // Calculate the error on the output layer
    for (size_t i = 0; i < n_; ++i) {
        float out = neurons.back()[i];
        float delta = target[i] - out;
        errors.back()[i] = out * (1 - out) * delta;
    }

    // Calculate errors for inner layers
    for (int l = m_ - 2; l >= 0; --l) {
        for (size_t j = 0; j < n_; ++j) {
            float error_next = errors[l + 1][j];
            for (size_t i = 0; i < n_; ++i) {
                errors[l][i] += weights[l][j][i] * error_next;
            }
        }

        for (size_t i = 0; i < n_; ++i) {
            float out = neurons[l][i];
            errors[l][i] *= out * (1 - out);
        }
    }

    // Update weights
    for (size_t l = 0; l < m_ - 1; ++l) {
        for (size_t i = 0; i < n_; ++i) {
            float error = errors[l + 1][i];
            for (size_t j = 0; j < n_; ++j) {
                weights[l][i][j] += eta_ * error * neurons[l][j];
            }
        }
    }
}

const std::vector<float>& NeuralMatrix::ReadOutput() const {
    return neurons.back();
}

void NeuralMatrix::SetLearningCoefficient(float eta) {
    eta_ = eta;
}

namespace tests {

void Propagate() {
    auto set_unit = [](std::vector<float>& vec, size_t pos) {
        std::fill(vec.begin(), vec.end(), 0.0f);
        vec[pos] = 1;
    };

    NeuralMatrix nm(3, 10);
    nm.InitializeWeightsWithRandom();
    nm.InitializeBiasesWithRandom();
    nm.SetLearningCoefficient(1.0f);

    // Propagate
    std::vector<float> resources(10);
    for (size_t t = 0; t < 3000; ++t) {
        set_unit(resources, t % 10);
        nm.CalculateOutput(resources);
        set_unit(resources, 9 - t % 10); // flip vector
        nm.PropagateErrorBack(resources);
    }

    // Check that the network has learned to flip vector
    for (size_t i = 0; i < 10; ++i) {
        set_unit(resources, i);
        nm.CalculateOutput(resources);
        const std::vector<float>& output = nm.ReadOutput();
        auto it = std::max_element(output.begin(), output.end());
        size_t index = it - output.begin();
        assert(index == 9 - i);
    }
}
}