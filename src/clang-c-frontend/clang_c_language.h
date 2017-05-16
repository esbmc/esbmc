/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CLANG_C_FRONTEND_CLANG_C_LANGUAGE_H_
#define CLANG_C_FRONTEND_CLANG_C_LANGUAGE_H_

#include <util/language.h>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

// Forward dec, to avoid bringing in clang headers
namespace clang {
  class ASTUnit;
}

class clang_c_languaget: public languaget
{
public:
  virtual bool preprocess(
    const std::string &path,
    std::ostream &outstream,
    message_handlert &message_handler);

  virtual bool parse(
    const std::string &path,
    message_handlert &message_handler);

  virtual bool final(
    contextt &context,
    message_handlert &message_handler);

  virtual bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler);

  std::string id() const { return "c"; }
  std::string description() const { return "C"; }

  virtual void show_parse(std::ostream &out);

  // conversion from expression into string
  virtual bool from_expr(
    const exprt &expr,
    std::string &code,
    const namespacet &ns);

  // conversion from type into string
  virtual bool from_type(
    const typet &type,
    std::string &code,
    const namespacet &ns);

  // conversion from string into expression
  virtual bool to_expr(
    const std::string &code,
    const std::string &module,
    exprt &expr,
    message_handlert &message_handler,
    const namespacet &ns);

  virtual languaget *new_language()
  { return new clang_c_languaget; }

  // constructor, destructor
  virtual ~clang_c_languaget() = default;
  clang_c_languaget();

protected:
  std::string internal_additions();

  void dump_clang_headers(std::string tmp_dir);
  void build_compiler_args(std::string tmp_dir);

  std::vector<std::string> compiler_args;
  std::vector<std::unique_ptr<clang::ASTUnit> > ASTs;
};

languaget *new_clang_c_language();

#endif
