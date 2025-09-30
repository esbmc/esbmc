#pragma once

#include <string>
#include <fstream>
#include <vector>

class module_locator
{
public:
  explicit module_locator(std::string output_dir);

  /**
    * @brief Computes the canonical file path for a qualified module name.
    *        Example: "a.b.c" -> "<out_dir>/a/b/c.ext".
    */
  std::string module_path(const std::string &qualified_module) const;

  /**
    * @brief Opens the module file for reading.
    *        Does not throw; callers should check `is_open()`.
    */
  std::ifstream open_module_file(const std::string &qualified_module) const;

private:
  static std::vector<std::string> split(const std::string &s, char delim);

  std::string out_dir_;
};
