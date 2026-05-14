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

void clang_cpp_languaget::force_file_type(
  std::vector<std::string> &compiler_args)
{
  // C++ standard
  assert(config.language.lid == language_idt::CPP);
  const std::string &cppstd = config.language.std;
  if (!cppstd.empty())
    compiler_args.emplace_back("-std=" + cppstd);

  // Force clang see all files as .cpp
  compiler_args.push_back("-x");
  compiler_args.push_back("c++");
}

void clang_cpp_languaget::build_include_args(
  std::vector<std::string> &compiler_args)
{
  std::string cppinc;
  bool do_inc = !config.options.get_bool_option("no-abstracted-cpp-includes") &&
                !config.options.get_bool_option("no-library");

  if (do_inc)
  {
    cppinc = esbmct::abstract_cpp_includes();
    log_debug("c++", "Adding CPP includes: {}", cppinc);
    // Let the cpp include "overtake" others.
    compiler_args.push_back("-isystem");
    compiler_args.push_back(cppinc);
    // Suppress system C++ standard library headers on all platforms so that
    // ESBMC's bundled OMs are the sole source of C++ standard-library
    // definitions.  Mixing the OMs with host libc++/libstdc++ headers
    // causes ambiguous-name errors (e.g. char_traits, istream) because the
    // OMs define names in namespace std while the host headers put them in
    // an inline namespace (std::__1 on libc++, std:: on libstdc++ but with
    // different ODR identity).
    // Users who need the host headers can pass --no-abstracted-cpp-includes.
    compiler_args.push_back("-nostdinc++");
  }

  clang_c_languaget::build_include_args(compiler_args);

  if (do_inc)
  {
    /* add include search paths for the default "library" models */
    compiler_args.push_back("-I" + cppinc + "/CUDA");
    compiler_args.push_back("-I" + cppinc + "/Qt");
    compiler_args.push_back("-I" + cppinc + "/Qt/QtCore");
  }
}

std::string clang_cpp_languaget::internal_additions()
{
  std::string intrinsics = R"(
# 1 "esbmc_intrinsics.hh" 1
extern "C" {
#pragma push_macro("_Bool")
#undef _Bool
#define _Bool bool
)";
  intrinsics.append(clang_c_languaget::internal_additions());
  intrinsics.append(R"(
void __ESBMC_throw_bad_cast(void);
#undef _Bool
#pragma pop_macro("_Bool")
})");

  return intrinsics;
}

void clang_cpp_languaget::set_language_version()
{
  const auto &ls =
    clang::LangStandard::getLangStandardForKind(AST->getLangOpts().LangStd);
#if LLVM_VERSION_MAJOR >= 17
  if (ls.isCPlusPlus26())
    config.language.cpp_std = cxx_stdt::cpp26;
  else if (ls.isCPlusPlus23())
    config.language.cpp_std = cxx_stdt::cpp23;
#else
  if (ls.isCPlusPlus2b())
    config.language.cpp_std = cxx_stdt::cpp23;
#endif
  else if (ls.isCPlusPlus20())
    config.language.cpp_std = cxx_stdt::cpp20;
  else if (ls.isCPlusPlus17())
    config.language.cpp_std = cxx_stdt::cpp17;
  else if (ls.isCPlusPlus14())
    config.language.cpp_std = cxx_stdt::cpp14;
  else if (ls.isCPlusPlus11())
    config.language.cpp_std = cxx_stdt::cpp11;
  else
    config.language.cpp_std = cxx_stdt::cpp98;
}

bool clang_cpp_languaget::typecheck(contextt &context, const std::string &)
{
  set_language_version();
  clang_cpp_convertert converter(context, AST, "C++");
  if (converter.convert())
    return true;

  clang_cpp_adjust adjuster(context);
  if (adjuster.adjust())
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
