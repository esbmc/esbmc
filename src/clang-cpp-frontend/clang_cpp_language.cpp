#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Frontend/ASTUnit.h>
CC_DIAGNOSTIC_POP()

#include <util/c_link.h>
#include <c2goto/cprover_library.h>
#include <clang-cpp-frontend/clang_cpp_main.h>
#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <clang-cpp-frontend/clang_cpp_language.h>
#include <util/cpp_expr2string.h>
#include <clang-cpp-frontend/esbmc_internal_cpp.h>
#include <regex>
#include <util/filesystem.h>
#include <fstream>

languaget *new_clang_cpp_language()
{
  return new clang_cpp_languaget;
}

void clang_cpp_languaget::force_file_type()
{
  // C++ standard
  std::string cppstd = config.options.get_option("cppstd");
  if (!cppstd.empty())
  {
    auto it = std::find(standards.begin(), standards.end(), cppstd);
    if (it == standards.end())
    {
      log_error("Invalid C++ standard: {}", cppstd);
      abort();
    }
  }

  std::string clangstd = cppstd.empty() ? "-std=c++03" : "-std=c++" + cppstd;
  compiler_args.push_back(clangstd);

  if (
    !config.options.get_bool_option("no-abstracted-cpp-includes") &&
    !config.options.get_bool_option("no-library"))
  {
    log_debug(
      "c++", "Adding CPP includes: {}", esbmct::abstract_cpp_includes());
    //compiler_args.push_back("-cxx-isystem");
    //compiler_args.push_back(esbmct::abstract_cpp_includes());
    // Let the cpp include "overtake" others.
    // Bear in mind that this is just a workaround to make sure we include the right headers we want,
    // and to get consistent error signatures in standalone runs and CIs
    compiler_args.push_back("-I" + esbmct::abstract_cpp_includes());
    compiler_args.push_back("-I" + esbmct::abstract_cpp_includes() + "/CUDA");
    compiler_args.push_back("-I" + esbmct::abstract_cpp_includes() + "/Qt");
    compiler_args.push_back(
      "-I" + esbmct::abstract_cpp_includes() + "/Qt/QtCore");
  }

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
  const std::string &module)
{
  contextt new_context;

  clang_cpp_convertert converter(new_context, ASTs, "C++");
  if (converter.convert())
    return true;

  clang_cpp_adjust adjuster(new_context);
  if (adjuster.adjust())
    return true;

  if (c_link(context, new_context, module))
    return true;

  return false;
}

bool clang_cpp_languaget::final(contextt &context)
{
  add_cprover_library(context);
  clang_cpp_maint cpp_main(context);
  return cpp_main.clang_main();
}

bool clang_cpp_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns,
  unsigned flags)
{
  code = cpp_expr2string(expr, ns, flags);
  return false;
}

bool clang_cpp_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns,
  unsigned flags)
{
  code = cpp_type2string(type, ns, flags);
  return false;
}
