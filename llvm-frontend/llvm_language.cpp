/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <llvm_language.h>
#include <llvm_parser.h>
#include <llvm_typecheck.h>

void llvm_languaget::modules_provided(std::set<std::string> &modules)
{
  modules.insert(parse_path);
}

bool llvm_languaget::preprocess(
  const std::string &path,
  std::ostream &outstream,
  message_handlert &message_handler)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

void llvm_languaget::internal_additions(std::ostream &out)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
}

bool llvm_languaget::parse(
  const std::string &path,
  message_handlert &message_handler)
{
  // store the path

  parse_path=path;

  llvm_parser.filename=path;

  bool res = llvm_parser.parse();

  ASTs.swap(llvm_parser.ASTs);

  return res;
}

bool llvm_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  contextt new_context;
  llvm_typecheckt typecheck(context);
  typecheck.ASTs.swap(ASTs);

  return typecheck.typecheck();
}

bool llvm_languaget::final(
  contextt &context,
  message_handlert &message_handler)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

void llvm_languaget::show_parse(std::ostream &out)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
}

languaget *new_llvm_language()
{
  return new llvm_languaget;
}

bool llvm_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

bool llvm_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

bool llvm_languaget::to_expr(
  const std::string &code,
  const std::string &module __attribute__((unused)),
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}
