#include "Utils.hpp"

namespace Sif{
namespace Utils{
    void debug_info(const std::string& _info) {
        if (std::getenv("SIF_DEBUG") != nullptr) {
            std::cerr << "[SIF_DEBUG_INFO]" << _info << "\n";
        }
    }

    // trim string from left side
    // from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
    void ltrim(std::string& _str) {
        _str.erase(_str.begin(), std::find_if(_str.begin(), _str.end(), [](int ch) {
        return !std::isspace(ch);
        }));
    }

    // trim string from right side
    // from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
    void rtrim(std::string& _str) {
        _str.erase(std::find_if(_str.rbegin(), _str.rend(), [](int ch) {
            return !std::isspace(ch);
        }).base(), _str.end());
    }

    // trim from both ends (in place)
    // from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
    void trim(std::string& _str) {
        ltrim(_str);
        rtrim(_str);
    }

    // trim from left side but no overwrite
    // from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
    std::string ltrim_copy(const std::string& _str) {
        return _str.substr(_str.find_first_not_of(' '));
    }

    // trim from right side but no overwrite
    // from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
    std::string rtrim_copy(const std::string& _str) {
        return _str.substr(0, _str.find_last_not_of(' ') + 1);
    }

    // trim from both sides but no overwrite
    // from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
    std::string trim_copy(const std::string& _str) {
        return _str.substr(_str.find_first_not_of(' '), _str.find_last_not_of(' ') - _str.find_first_not_of(' ') + 1);
    }

    //from https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
    std::vector<std::string> split(const std::string& _str, const std::string& _delimiter) {
        std::vector<std::string> result;
        std::string str = std::string(_str);
        size_t pos = 0;
        std::string token;
        while ((pos = str.find(_delimiter)) != std::string::npos) {
            token = str.substr(0, pos);
            result.push_back(token);
            str.erase(0, pos + _delimiter.length());
        }
        result.push_back(str);
        return result;
    }

    std::string retrieve_string_element(const std::string& _str, const unsigned int& _index, const std::string& _delimiter) {
        std::string new_str= trim_copy(_str);
        std::vector<std::string> v_str = split(new_str, _delimiter);
        if (_index < v_str.size()) {
            return v_str[_index];
        } else {
            return "";
        }
    }

    std::string substr_by_edge(const std::string& _str, const std::string& _left, const std::string& _right) {
        auto left = _str.find(_left);
        auto right = _str.rfind(_right);
        if (left == std::string::npos || right == std::string::npos) {
            return "";
        } else {
            return _str.substr(_left.length() + left, right - _left.length() - left);
        }
    }

    //from https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
    void str_replace_all(std::string& _str, const std::string& _from, const std::string& _to) {
        if(_from.empty()) {
            return;
        }
        size_t start_pos = 0;
        while((start_pos = _str.find(_from, start_pos)) != std::string::npos) {
            _str.replace(start_pos, _from.length(), _to);
            start_pos += _to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
        }
    }

    std::map<std::string, std::string> parse_visitor_args(const std::string& _args) {
        auto values = split(_args, " ");
        std::map<std::string, std::string> result;
        for (auto it = values.begin(); it != values.end(); ++it) {
            auto pair = split(*it, "=");
            if (pair.size() != 2) {
                continue;
            }
            result[pair[0]] = pair[1];
        }
        return result;
    }

}
}
