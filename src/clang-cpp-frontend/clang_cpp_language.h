/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_LANGUAGE_H
#define CPROVER_CPP_LANGUAGE_H

#include <util/language.h>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

class clang_cpp_languaget : public languaget
{
public:
  virtual bool preprocess(
    const std::string &path,
    std::ostream &outstream,
    message_handlert &message_handler);

  bool
  parse(const std::string &path, message_handlert &message_handler) override;

  bool final(contextt &context, message_handlert &message_handler) override;

  bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler) override;

  bool convert(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler);

  void show_parse(std::ostream &out) override;

  // conversion from expression into string
  bool from_expr(
    const exprt &expr,
    std::string &code,
    const namespacet &ns,
    bool fullname = false) override;

  // conversion from type into string
  bool from_type(
    const typet &type,
    std::string &code,
    const namespacet &ns,
    bool fullname = false) override;

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
