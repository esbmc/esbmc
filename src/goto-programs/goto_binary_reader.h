#pragma once

#include <util/symtab/context.h>
#include <vector>
#include <string>

class context;
class goto_functionst;

class goto_binary_reader
{
public:
  /** Reads a bundled library blob. \p goto_functions is null for a blob whose
   *  bodies travel in their symbols' values and are built later by
   *  goto_convert_functions; pass one for a blob written after goto_convert,
   *  whose bodies are already lowered. */
  bool read_goto_binary_array(
    const void *data,
    size_t size,
    contextt &context,
    contextt &ignored,
    goto_functionst *goto_functions = nullptr);

  void set_functions_to_read(const std::vector<std::string> &funcs)
  {
    function_set.insert(funcs.begin(), funcs.end());
  }

  bool read_goto_binary(
    const std::string &path,
    contextt &context,
    goto_functionst &dest);

private:
  // whitelist (if not empty) of functions to read symbols of
  std::unordered_set<std::string> function_set;
};
