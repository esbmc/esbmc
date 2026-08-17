#pragma once

#include <util/symtab/context.h>
#include <vector>
#include <string>

class context;
class goto_functionst;

class goto_binary_reader
{
public:
  /** Reads the symbol table only; the bundled library blob has no bodies. */
  bool read_goto_binary_array(
    const void *data,
    size_t size,
    contextt &context,
    contextt &ignored);

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
