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

class file_operations
{
 public:
  /**
   * @brief Generates a unique path based on the format
   *
   * In Linux, running this function with "esbmc-%%%%" will
   * return a string such as "/tmp/esbmc-0001" or "/tmp/esbmc-8787".
   *
   * This function does not have garantee that will finish
   * and can be run forever until it sees an available spot.
   *
   * @param format A string in the file specification
   */
  static std::string get_unique_tmp_path(const char *format);
};
