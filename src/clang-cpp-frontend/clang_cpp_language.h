/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_LANGUAGE_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_LANGUAGE_H_

#include <clang-c-frontend/clang_c_language.h>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

// Forward dec, to avoid bringing in clang headers
namespace clang {
  class ASTUnit;
} // namespace clang

class clang_cpp_languaget: public clang_c_languaget
{
public:
  std::string id() const override { return "cpp"; }

  bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler) override ;

  // conversion from expression into string
  bool from_expr(
    const exprt &expr,
    std::string &code,
    const namespacet &ns,
    bool fullname = false) override ;

  // conversion from type into string
  bool from_type(
    const typet &type,
    std::string &code,
    const namespacet &ns,
    bool fullname = false) override ;

  languaget *new_language() override
  { return new clang_cpp_languaget; }

  // constructor, destructor
  ~clang_cpp_languaget() override = default;
  clang_cpp_languaget();

protected:
  std::string internal_additions() override;
  void force_file_type() override;
};

languaget *new_clang_cpp_language();

#endif
