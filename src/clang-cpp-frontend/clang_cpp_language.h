/*******************************************************************\

Module: C++ Language Module

Author:

\*******************************************************************/

#ifndef CPROVER_CPP_LANGUAGE_H
#define CPROVER_CPP_LANGUAGE_H

#include <clang-c-frontend/clang_c_language.h>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

class clang_cpp_languaget : public clang_c_languaget
{
public:
  std::string id() const override
  {
    return "cpp";
  }

  languaget *new_language() override
  {
    return new clang_cpp_languaget;
  }

  // constructor, destructor
  ~clang_cpp_languaget() override = default;
  clang_cpp_languaget();
};

languaget *new_clang_cpp_language();

#endif
