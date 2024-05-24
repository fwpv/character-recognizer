#pragma once

#include <vector>

class NeuralMatrix {
public:
    /// NOTE: It doesn't work very well for m > 5,
    /// because it requires a very large number of training cycles

    NeuralMatrix(size_t m, size_t n);
    void InitializeWeightsWithRandom();
    void InitializeBiasesWithRandom(float min = 0.0f, float max = 0.1f);

    void CalculateOutput(const std::vector<float>& input) noexcept;
    float EvaluateError(const std::vector<float>& target, size_t size) const;
    float EvaluateError(const std::vector<float>& target) const;
    void PropagateErrorBack(const std::vector<float>& target) noexcept;
    const std::vector<float>& ReadOutput() const;

    void SetLearningCoefficient(float eta);

private:
    // first index - layer number
    // second index - neuron number
    // value - output [0..1]
    std::vector<std::vector<float>> neurons;
    
    // first index - interlayer number (interlayers = m - 1)
    // third index - neuron of a neuron in current layer
    // second index - neuron of a neuron in previous layer
    // value - weight [-1..1]
    std::vector<std::vector<std::vector<float>>> weights;

    // first index - layer number
    // second index - neuron number
    // value - error
    std::vector<std::vector<float>> errors;

    // first index - layer number (layers = m - 1)
    // value - bias [0..1]
    std::vector<float> biases;

    size_t m_; // number of layers
    size_t n_; // number of inputs and outputs
    float eta_ = 0.5f; // learning coefficient [0..1]
};

namespace tests {

void Propagate();

}
