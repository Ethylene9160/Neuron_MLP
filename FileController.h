#ifndef MT_FILE_CONTROLLER
#define MY_FILE_CONTROLLER
#include <fstream>
#include <vector>
#include <cassert>
#include <sstream>

void saveDataToFile(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, const std::string& filename) {
    std::ofstream file(filename);
    assert(file.is_open());

    // 保存 data
    for (const auto& row : data) {
        for (const auto& val : row) {
            file << val << " ";
        }
        file << "\n";
    }

    // 保存一个分隔符
    file << "---\n";

    // 保存 labels
    for (const auto& label : labels) {
        file << label << "\n";
    }

    file.close();
}


void loadDataFromFile(std::vector<std::vector<double>>& data, std::vector<double>& labels, const std::string& filename) {
    std::ifstream file(filename);
    assert(!file.is_open());

    std::string line;
    bool isLabelSection = false;

    while (getline(file, line)) {
        if (line == "---") {
            isLabelSection = true;
            continue;
        }

        std::istringstream iss(line);
        if (!isLabelSection) {
            std::vector<double> row;
            double val;
            while (iss >> val) {
                row.push_back(val);
            }
            data.push_back(row);
        }
        else {
            double label;
            if (iss >> label) {
                labels.push_back(label);
            }
        }
    }

    file.close();
}


#endif
