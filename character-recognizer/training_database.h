#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

class FileNormalizerInterface {
public:
    virtual std::vector<float> Load(const std::filesystem::path& file) const = 0;
};

class ImageFileNormalizer : public FileNormalizerInterface {
public:
    ImageFileNormalizer(size_t input_width);
    std::vector<float> Load(const std::filesystem::path& file) const override;

private:
    size_t input_width_;
};

// Database for training neural networks
// It loads images and stores them as normalized values
// for neural network input

class TrainingDatabase {
public:
    using Data = std::vector<std::vector<float>>;
    using DataDict = std::unordered_map<char, Data>;

    TrainingDatabase(const FileNormalizerInterface* file_normalizer);
    void BuildFromFolder(const std::filesystem::path& folder);

    const DataDict& GetDataDictionary() const;
    const Data& GetNonCharData() const;
    std::vector<char> GetUploadedChars() const;

private:
    const FileNormalizerInterface* file_normalizer_;
    DataDict data_dict_;
    Data non_char_data_;
};