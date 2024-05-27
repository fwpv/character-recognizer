#include "snn.h"
#include "profiler.h"
#include "training.h"
#include "training_database.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std::filesystem;
using namespace std::literals;

void RunExperiment1() {
    // Load data
    ImageFileNormalizer normalizer(1024);
    TrainingDatabase db(&normalizer);
    db.BuildFromFolder(path("numbers"s));

    // Init neural matrix
    SNN snn(1024, 2, 1024, 10);
    snn.InitializeWeightsWithRandom();
    snn.InitializeBiasesWithRandom();
    snn.SetLearningCoefficient(0.01f);

    auto set_unit = [](std::vector<float>& vec, size_t pos) {
        std::fill(vec.begin(), vec.end(), 0.0f);
        vec[pos] = 1;
    };

    auto clear = [](std::vector<float>& vec) {
        std::fill(vec.begin(), vec.end(), 0.0f);
    };

    std::vector<float> resources(10, 0.0f);

    // Training
    {
        LOG_DURATION("Training"s);
        const TrainingDatabase::DataDict& dict = db.GetDataDictionary();
        const TrainingDatabase::Data& not_chars = db.GetNonCharData();
        for (auto& it : dict) {
            char c = it.first;
            if (c < '0' || c > '9') {
                throw std::runtime_error("Char "s + c + " is not supported"s);
            }
        }

        for (size_t i = 0; i < 1000; ++i) {
            for (auto& it : dict) {
                char c = it.first;
                size_t number = c - '0';
                for (auto& vec : it.second) {
                    snn.CalculateOutput(vec);
                    set_unit(resources, number);
                    snn.PropagateErrorBack(resources);
                    auto& ns_vec = not_chars[rand() % not_chars.size()];
                    snn.CalculateOutput(ns_vec);
                    clear(resources);
                    snn.PropagateErrorBack(resources);
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
            snn.CalculateOutput(vec);
            const std::vector<float>& output = snn.ReadOutput();
            auto it = std::max_element(output.begin(), output.end());
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
        snn.CalculateOutput(vec);
        const std::vector<float>& output = snn.ReadOutput();
        auto it = std::max_element(output.begin(), output.end());
        size_t max = it - output.begin();
        std::cout << c << ": " << max << "  p: " << *it << std::endl;
    }
}