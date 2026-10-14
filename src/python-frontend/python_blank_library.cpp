#include <python-frontend/python_library.h>

#include <goto-programs/goto_functions.h>

// python2goto produces the blob this would load, so it cannot depend on it.
// The C analogue is src/c2goto/cprover_blank_library.cpp.
void add_cpython_library(contextt &)
{
}

const goto_functionst &cpython_library_bodies()
{
  static const goto_functionst empty;
  return empty;
}
