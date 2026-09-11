#include <util/base/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Frontend/ASTUnit.h>
CC_DIAGNOSTIC_POP()

#include <util/lang/c_link.h>
#include <c2goto/cprover_library.h>
#include <clang-cpp-frontend/clang_cpp_main.h>
#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <set>
#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <clang-cpp-frontend/clang_cpp_language.h>
#include <util/lang/cpp_expr2string.h>
#include <clang-cpp-frontend/esbmc_internal_cpp.h>
#include <regex>
#include <util/base/filesystem.h>
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

  /* Clang gives std::addressof a BuiltinAttr and rewrites calls to it, which
   * discards the operational model's definition and leaves symex a body-less
   * function returning a nondet pointer (github #6063). We need the real
   * body, so opt out of the builtin. */
  compiler_args.emplace_back("-fno-builtin-std-addressof");
}

void clang_cpp_languaget::build_include_args(
  std::vector<std::string> &compiler_args)
{
  std::string cppinc;
  bool do_inc = !config.options.get_bool_option("no-abstracted-cpp-includes") &&
                !config.options.get_bool_option("no-library");

  if (!do_inc && config.options.get_bool_option("mix-cpp-host-headers"))
    log_warning(
      "--mix-cpp-host-headers has no effect: the abstracted C++ includes "
      "are already disabled via --no-abstracted-cpp-includes or "
      "--no-library, so only the host headers are used");

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
    // Users who need only the host headers can pass
    // --no-abstracted-cpp-includes; users who want both side by side (e.g.
    // to reach a system header the bundled OMs don't cover, accepting the
    // ambiguous-name risk above) can pass --mix-cpp-host-headers instead.
    if (!config.options.get_bool_option("mix-cpp-host-headers"))
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

// Exception-lowering runtime hooks. remove_exceptions inserts calls to these
// after parsing, so they must be visible during library linking for the OM
// bodies to be pulled into the goto program. Declared only for C++ (the C
// frontend cannot produce throw/catch): a pure-C program never uses exception
// lowering, so injecting these into every C TU would force the OM bodies to be
// linked into exception-free programs, perturbing analyses such as
// termination's recurrent-set search.
void __ESBMC_push_handled_exception(void);
void __ESBMC_pop_handled_exception(void);
void __ESBMC_rethrow_current_exception(void);
void *__ESBMC_current_exception_raw(void);
void __ESBMC_rethrow_exception_raw(void *);
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

/// Phase 7 census: force migrate_type/migrate_expr over every symbol this TU
/// contributed and discard the result, to find what the C++ frontend emits that
/// IREP2 cannot represent. Runs after c_link so migrate_namespace_lookup can
/// resolve the TU's own symbols: before the link they are all absent from it,
/// and sym_name_to_symbol then substitutes the expression's own type for
/// migrate_symbol_type's, which is the case migrate.cpp warns hashes wrongly.
///
/// Migration reports failure by throwing a std::string, so each symbol is
/// wrapped: one unrepresentable construct names itself and the walk continues,
/// which is what makes this a census rather than a bisection.
///
/// Walks the whole linked context, operational models included, which is the
/// set goto_convert migrates anyway. On a multi-TU run the later counts
/// therefore include the earlier TUs' symbols.
/// Rejected alternatives, and what it measured: scope-clang-cpp-irep2.md §4.
static void migrate_census(const contextt &context)
{
  unsigned long symbols = 0, values = 0, failures = 0;
  // The kind tally is what stops the census being vacuous: a count of symbols
  // or values is identical on either representation, so swapping get_type2()
  // for get_type() would migrate nothing and print the same line. A type_id
  // exists only on the IREP2 side.
  std::set<unsigned> kinds;
  context.foreach_operand_in_order(
    [&symbols, &values, &failures, &kinds](const symbolt &s) {
      ++symbols;
      try
      {
        kinds.insert(static_cast<unsigned>(s.get_type2()->type_id));
        if (!is_nil_expr(s.get_value2()))
          ++values;
      }
      catch (const std::string &e)
      {
        ++failures;
        log_error("IREP2 migrate census: {} on symbol {}", e, s.id);
      }
    });
  log_status(
    "IREP2 migrate census: {} symbols, {} values migrated, {} type kinds, {} "
    "failures",
    symbols,
    values,
    kinds.size(),
    failures);
}

bool clang_cpp_languaget::typecheck(
  contextt &context,
  const std::string &module)
{
  set_language_version();

  // Convert + adjust this translation unit in an isolated context, then
  // merge the fully-adjusted symbols into the shared context via c_link.
  // This keeps each TU's convert/adjust from re-walking and re-adjusting
  // symbols contributed by other frontends/TUs sharing `context` (#5309).
  contextt new_context;

  clang_cpp_convertert converter(new_context, AST, "C++");
  if (converter.convert())
    return true;

  clang_cpp_adjust adjuster(new_context);
  if (adjuster.adjust())
    return true;

  if (c_link(context, new_context, module))
    return true;

  if (config.options.get_bool_option("clang-cpp-irep2-migrate-census"))
    migrate_census(context);

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
