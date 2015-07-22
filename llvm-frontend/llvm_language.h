/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_LANGUAGE_H
#define CPROVER_CPP_LANGUAGE_H

#include <language.h>

class llvm_languaget:public languaget
{
public:
  virtual bool preprocess(
    const std::string &path,
    std::ostream &outstream,
    message_handlert &message_handler);

  virtual bool parse(
    const std::string &path,
    message_handlert &message_handler);

  virtual bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler);

  virtual bool final(
    contextt &context,
    message_handlert &message_handler);

  virtual void show_parse(std::ostream &out);

  // constructor, destructor
  virtual ~llvm_languaget() { };
  llvm_languaget() { }

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
  { return new llvm_languaget; }

  virtual std::string id() const { return "c"; }
  virtual std::string description() const { return "C"; }

  virtual void modules_provided(std::set<std::string> &modules);

protected:
  std::string parse_path;

  void internal_additions(std::ostream &outstream);

  virtual std::string main_symbol()
  {
    return "c::main";
  }
};

languaget *new_llvm_language();

#endif
