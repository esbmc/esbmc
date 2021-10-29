/*******************************************************************\

Module: File operations

Author: Rafael Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/

#pragma once

#include <string>
/**
 * @brief this file will contains helper functions for manipulating
          files
 */

namespace file_operations {

/**
 * @brief Represents a temporary directory, which is removed by the destructor.
 */
class tmp_dir
{
  std::string _path;
  bool _keep;

public:
  tmp_dir(std::string path, bool keep = false);
  ~tmp_dir();

  tmp_dir(const tmp_dir &) = delete;

  tmp_dir & operator=(const tmp_dir &) = delete;

  const std::string &path() const noexcept
  {
    return _path;
  }

  tmp_dir &keep(bool yes) &noexcept
  {
    _keep = yes;
    return *this;
  }

  tmp_dir &&keep(bool yes) &&noexcept
  {
    _keep = yes;
    return std::move(*this);
  }
};

/**
 * @brief Generates a unique path based on the format
 *
 * In Linux, running this function with "esbmc-%%%%" will
 * return a string such as "/tmp/esbmc-0001" or "/tmp/esbmc-8787".
 *
 * This function does not have guarantee that will finish
 * and can be run forever until it sees an available spot.
 *
 * @param format A string in the file specification
 */
const std::string get_unique_tmp_path(const std::string &format);

}
