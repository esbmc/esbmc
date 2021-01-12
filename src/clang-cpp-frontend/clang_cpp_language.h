/*******************************************************************\

Module: Clang C++ Language Module

Author:

\*******************************************************************/

#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_LANGUAGE_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_LANGUAGE_H_

#include <clang-c-frontend/clang_c_language.h>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

class clang_cpp_languaget : public clang_c_languaget
{
public:
  bool final(contextt &context, message_handlert &message_handler) override;

  bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler) override;

  std::string id() const override
  {
    return "cpp";
  }

  // conversion from expression into string
  bool from_expr(const exprt &expr, std::string &code, const namespacet &ns)
    override;

  // conversion from type into string
  bool from_type(const typet &type, std::string &code, const namespacet &ns)
    override;

  languaget *new_language() override
  {
    return new clang_cpp_languaget;
  }

  // constructor, destructor
  ~clang_cpp_languaget() override = default;
  clang_cpp_languaget();

protected:
  std::string internal_additions() override;
  void force_file_type() override;
};

languaget *new_clang_cpp_language();

#endif
