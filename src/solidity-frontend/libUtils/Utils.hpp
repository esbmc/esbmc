#ifndef SIF_LIBUTILS_UTILS_H_
#define SIF_LIBUTILS_UTILS_H_

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace Sif{
namespace Utils{
    void debug_info(const std::string& _info);

    // trim string from left side
    void ltrim(std::string& _str);

    // trim string from right side
    void rtrim(std::string& _str);

    // trim from both ends (in place)
    void trim(std::string& _str);

    // trim from left side but no overwrite
    std::string ltrim_copy(const std::string& _str);

    // trim from right side but no overwrite
    std::string rtrim_copy(const std::string& _str);

    // trim from both sides but no overwrite
    std::string trim_copy(const std::string& _str);

    std::vector<std::string> split(const std::string& _str, const std::string& _delimiter);

    // Get the _index[th] element from a string split by _c
    std::string retrieve_string_element(const std::string& _str, const unsigned int& _index, const std::string& _delimiter);

    std::string substr_by_edge(const std::string& _str, const std::string& _left, const std::string& _right);

    //from https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
    void str_replace_all(std::string& _str, const std::string& _from, const std::string& _to);

    std::map<std::string, std::string> parse_visitor_args(const std::string& _args);
}
}

#endif //SIF_LIBUTILS_UTILS_H_