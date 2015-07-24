/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_LANGUAGE_H
#define CPROVER_CPP_LANGUAGE_H

#include <language.h>

#include <clang/Frontend/ASTUnit.h>

#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

class llvm_languaget
{
public:
  bool preprocess(
    const std::string &path,
    std::ostream &outstream,
    message_handlert &message_handler);

  bool parse();

  bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler);

  // constructor, destructor
  virtual ~llvm_languaget();
  llvm_languaget(std::vector<std::string> _files);

  // conversion from expression into string
  bool from_expr(
    const exprt &expr,
    std::string &code,
    const namespacet &ns);

  // conversion from type into string
  bool from_type(
    const typet &type,
    std::string &code,
    const namespacet &ns);

  // conversion from string into expression
  bool to_expr(
    const std::string &code,
    const std::string &module,
    exprt &expr,
    message_handlert &message_handler,
    const namespacet &ns);

  std::string id() const { return "c"; }
  std::string description() const { return "C"; }

protected:
  std::vector<std::string> files;

  void internal_additions(std::ostream &outstream);

  virtual std::string main_symbol()
  {
    return "c::main";
  }

  std::vector<std::unique_ptr<clang::ASTUnit> > ASTs;
};

#endif
