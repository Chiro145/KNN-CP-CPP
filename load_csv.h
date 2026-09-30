// This lib gen by Gemini cuz I suck

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

inline bool load_train_kaggle(const std::string& filepath, std::vector<Sample>& train_set) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Khong the mo file: " << filepath << "\n";
        return false;
    }

    std::string line;
    std::getline(file, line);
    train_set.reserve(42000);

    Sample s;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        const char* p = line.c_str();

        int val = 0;
        while (*p >= '0' && *p <= '9') {
            val = val * 10 + (*p - '0');
            p++;
        }
        s.label = static_cast<uint8_t>(val);

        for (int i = 0; i < 784; ++i) {
            if (*p == ',') p++;
            val = 0;
            while (*p >= '0' && *p <= '9') {
                val = val * 10 + (*p - '0');
                p++;
            }
            s.pixel[i] = static_cast<uint8_t>(val);
        }
        train_set.push_back(s);
    }
    return true;
}

inline bool load_test_kaggle(const std::string& filepath, std::vector<Sample>& test_set) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Khong the mo file: " << filepath << "\n";
        return false;
    }

    std::string line;
    std::getline(file, line);
    test_set.reserve(28000);

    Sample s;
    s.label = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        const char* p = line.c_str();

        for (int i = 0; i < 784; ++i) {
            if (i > 0 && *p == ',') p++;
            int val = 0;
            while (*p >= '0' && *p <= '9') {
                val = val * 10 + (*p - '0');
                p++;
            }
            s.pixel[i] = static_cast<uint8_t>(val);
        }
        test_set.push_back(s);
    }
    return true;
}

inline void export_submission(const std::string& out_path, const std::vector<uint8_t>& preds) {
    std::ofstream out(out_path);
    out << "ImageId,Label\n";
    for (size_t i = 0; i < preds.size(); ++i) {
        out << (i + 1) << "," << static_cast<int>(preds[i]) << "\n";
    }
}
