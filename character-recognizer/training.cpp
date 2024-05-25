#include "neural_matrix.h"
#include "profiler.h"
#include "training.h"
#include "training_database.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std::filesystem;
using namespace std::literals;

void RunExperiment1() {
    // Load data
    ImageFileNormalizer normalizer(1024);
    TrainingDatabase db(&normalizer);
    db.BuildFromFolder(path("numbers"s));

    // Init neural matrix
    NeuralMatrix nm(4, 1024);
    nm.InitializeWeightsWithRandom();
    nm.SetLearningCoefficient(0.01f);

    auto set_unit = [](std::vector<float>& vec, size_t pos) {
        std::fill(vec.begin(), vec.begin() + 10, 0.0f);
        vec[pos] = 1;
    };

    std::vector<float> resources(1024, 0.0f);

    // Training
    {
        LOG_DURATION("Training"s);
        const TrainingDatabase::DataDict& dict = db.GetDataDictionary();
        for (auto& it : dict) {
            char c = it.first;
            if (c < '0' || c > '9') {
                throw std::runtime_error("Char "s + c + " is not supported"s);
            }
        }

        for (size_t i = 0; i < 5000; ++i) {
            for (auto& it : dict) {
                char c = it.first;
                size_t number = c - '0';
                for (auto& vec : it.second) {
                    nm.CalculateOutput(vec);
                    set_unit(resources, number);
                    nm.PropagateErrorBack(resources);
                }
            }
        }
    }

    // Check result
    const TrainingDatabase::DataDict& dict = db.GetDataDictionary();
    std::cout << std::fixed << std::showpoint << std::setprecision(2);
    for (int c = 0; c < 10; ++c) {
        std::cout << c << ":"s;
        std::vector<float> precisions;
        for (int k = 0; k < 10; ++k) {
            const auto& vec = dict.at('0' + c)[k];
            nm.CalculateOutput(vec);
            const std::vector<float>& output = nm.ReadOutput();
            auto it = std::max_element(output.begin(), output.begin() + 10);
            size_t max = it - output.begin();
            std::cout << ' ' << max;
            precisions.push_back(*it);
        }
        std::cout << " p: "s;
        for (int p = 0; p < 10; ++p) {
            std::cout << ' ' << precisions[p];
        }
        std::cout << std::endl;
    }

    // Test img
    std::cout << "Test img" << std::endl;
    std::cout << std::fixed << std::showpoint << std::setprecision(5);
    for (int c = 0; c < 10; ++c) {
        std::string i_str = std::to_string(c);
        auto vec = normalizer.Load(path("test_img"s) / (i_str + ".bmp"s));
        nm.CalculateOutput(vec);
        const std::vector<float>& output = nm.ReadOutput();
        auto it = std::max_element(output.begin(), output.begin() + 10);
        size_t max = it - output.begin();
        std::cout << c << ": " << max << "  p: " << *it << std::endl;
    }
}