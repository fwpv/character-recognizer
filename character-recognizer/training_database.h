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
    using Char = std::vector<float>;
    using Chars = std::vector<Char>;
    using CharsDict = std::unordered_map<char, Chars>;
    using CharPtrArray = std::vector<std::pair<char, const Char*>>;

    TrainingDatabase(const FileNormalizerInterface* file_normalizer);
    void BuildFromFolder(const std::filesystem::path& folder);

    const CharsDict& GetCharsDictionary() const;
    const Chars& GetNonChars() const;
    CharPtrArray CreateCharPtrArray() const;

    static void ShuffleCharPtrArray(CharPtrArray& array);

private:
    const FileNormalizerInterface* file_normalizer_;
    CharsDict data_dict_;
    Chars non_chars_;
};