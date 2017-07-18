/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <ansi-c/c_link.h>
#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <clang-cpp-frontend/clang_cpp_language.h>
#include <clang-cpp-frontend/expr2cpp.h>
#include <regex>

languaget *new_clang_cpp_language()
{
  return new clang_cpp_languaget;
}

clang_cpp_languaget::clang_cpp_languaget()
  : clang_c_languaget()
{
}

void clang_cpp_languaget::force_file_type()
{
  // Force clang see all files as .cpp
  // This forces the preprocessor to be called even in preprocessed files
  // which allow us to perform transformations using -D
  compiler_args.emplace_back("-x");
  compiler_args.emplace_back("c++");

  // We also force the standard to be c++98
  compiler_args.push_back("-std=c++98");
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
  contextt& context,
  const std::string& module,
  message_handlert& message_handler)
{
  contextt new_context;

  clang_cpp_convertert converter(new_context, ASTs);
  if(converter.convert())
    return true;

  clang_cpp_adjust adjuster(new_context);
  if(adjuster.adjust())
    return true;

  if(c_link(context, new_context, message_handler, module))
    return true;

  // Remove unused
  if(!config.options.get_bool_option("keep-unused"))
    context.remove_unused();

  return false;
}

bool clang_cpp_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns,
  bool fullname)
{
  code=expr2cpp(expr, ns, fullname);
  return false;
}

bool clang_cpp_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns,
  bool fullname)
{
  code=type2cpp(type, ns, fullname);
  return false;
}
