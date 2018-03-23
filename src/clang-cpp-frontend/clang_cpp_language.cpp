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

clang_cpp_languaget::clang_cpp_languaget()
{
  std::cout
    << "ESBMC currently does not support parsing C++ programs using clang"
    << std::endl;
  abort();
}

bool clang_cpp_languaget::parse(
  const std::string &path __attribute__((unused)),
  message_handlert &message_handler __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet"
            << std::endl;
  abort();
  return true;
}

bool clang_cpp_languaget::typecheck(
  contextt &context __attribute__((unused)),
  const std::string &module __attribute__((unused)),
  message_handlert &message_handler __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet"
            << std::endl;
  abort();
  return true;
}

void clang_cpp_languaget::show_parse(std::ostream &out __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet"
            << std::endl;
  abort();
}

bool clang_cpp_languaget::convert(
  contextt &context __attribute__((unused)),
  const std::string &module __attribute__((unused)),
  message_handlert &message_handler __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet"
            << std::endl;
  abort();
  return true;
}

bool clang_cpp_languaget::preprocess(
  const std::string &path __attribute__((unused)),
  std::ostream &outstream __attribute__((unused)),
  message_handlert &message_handler __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet"
            << std::endl;
  abort();
  return true;
}

bool clang_cpp_languaget::final(
  contextt &context __attribute__((unused)),
  message_handlert &message_handler __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet"
            << std::endl;
  abort();
  return true;
}

bool clang_cpp_languaget::from_expr(
  const exprt &expr __attribute__((unused)),
  std::string &code __attribute__((unused)),
  const namespacet &ns __attribute__((unused)),
  bool fullname __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet"
            << std::endl;
  abort();
  return true;
}

bool clang_cpp_languaget::from_type(
  const typet &type __attribute__((unused)),
  std::string &code __attribute__((unused)),
  const namespacet &ns __attribute__((unused)),
  bool fullname __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet"
            << std::endl;
  abort();
  return true;
}
