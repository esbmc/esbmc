/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

// Remove warnings from Clang headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <clang/Frontend/ASTUnit.h>
#pragma GCC diagnostic pop

#include <util/c_link.h>
#include <c2goto/cprover_library.h>
#include <clang-c-frontend/clang_c_main.h>
#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <clang-cpp-frontend/clang_cpp_language.h>
#include <clang-cpp-frontend/expr2cpp.h>
#include <regex>

languaget *new_clang_cpp_language(const messaget &msg)
{
  return new clang_cpp_languaget(msg);
}

clang_cpp_languaget::clang_cpp_languaget(const messaget &msg)
  : clang_c_languaget(msg)
{
}

void clang_cpp_languaget::force_file_type()
{
  // We also force the standard to be c++98
  compiler_args.push_back("-std=c++98");

  // Force clang see all files as .cpp
  compiler_args.push_back("-x");
  compiler_args.push_back("c++");
}

std::string clang_cpp_languaget::internal_additions()
{
  std::string intrinsics = "extern \"C\" {\n";
  intrinsics.append(clang_c_languaget::internal_additions());
  intrinsics.append("}\n");

  // Replace _Bool by bool and return
  return std::regex_replace(intrinsics, std::regex("_Bool"), "bool");
}

bool clang_cpp_languaget::typecheck(
  contextt &context,
  const std::string &module,
  const messaget &message_handler)
{
  contextt new_context(message_handler);

  clang_cpp_convertert converter(new_context, ASTs, message_handler);
  if(converter.convert())
    return true;

  clang_cpp_adjust adjuster(new_context, message_handler);
  if(adjuster.adjust())
    return true;

  if(c_link(context, new_context, message_handler, module))
    return true;

  return false;
}

bool clang_cpp_languaget::final(
  contextt &context,
  const messaget &message_handler)
{
  add_cprover_library(context, message_handler);
  return clang_main(context, message_handler);
}

bool clang_cpp_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  code = expr2cpp(expr, ns);
  return false;
}

bool clang_cpp_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  code = type2cpp(type, ns);
  return false;
}
