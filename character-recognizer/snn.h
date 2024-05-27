#pragma once

#include <vector>

// Simple neural network
class SNN {
public:
    /// NOTE: It doesn't work very well for h_l > 3,
    /// because it requires a very large number of training cycles

    SNN(size_t i_n, size_t h_l, size_t h_n, size_t o_n);
    void InitializeWeightsWithRandom();
    void InitializeBiasesWithRandom(float min = 0.0f, float max = 0.1f);

    void CalculateOutput(const std::vector<float>& input) noexcept;
    float EvaluateError(const std::vector<float>& target) const;
    void PropagateErrorBack(const std::vector<float>& target) noexcept;
    const std::vector<float>& ReadOutput() const;

    void SetLearningCoefficient(float eta);

private:
    // Layers
    std::vector<float> input_layer;
    std::vector<std::vector<float>> hidden_layers;
    std::vector<float> output_layer;
    
    // Weights between layers
    std::vector<std::vector<std::vector<float>>> weights;

    // Biases for each hidden and output layer
    std::vector<std::vector<float>> biases;

    // Errors for each hidden and output layer
    std::vector<std::vector<float>> errors;

    // Network parameters
    size_t i_n_; // number of input neurons
    size_t h_l_; // number of hidden layers
    size_t h_n_; // number of neurons in hidden layer
    size_t o_n_; // number of output neurons
    float eta_ = 0.5f; // learning coefficient [0..1]
};

namespace tests {

void Propagate();

}
