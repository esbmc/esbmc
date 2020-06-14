/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <c2goto/cprover_library.h>
#include <clang-cpp-frontend/clang_cpp_language.h>

languaget *new_clang_cpp_language()
{
  return new clang_cpp_languaget;
}

clang_cpp_languaget::clang_cpp_languaget() : clang_c_languaget()
{
}
