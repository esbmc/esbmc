/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <AST/build_ast.h>
#include <ansi-c/c_preprocess.h>
#include <boost/filesystem.hpp>
#include <c2goto/cprover_library.h>
#include <clang-c-frontend/clang_c_adjust.h>
#include <clang-c-frontend/clang_c_convert.h>
#include <clang-c-frontend/clang_c_language.h>
#include <clang-c-frontend/clang_c_main.h>
#include <clang-c-frontend/expr2c.h>
#include <sstream>
#include <util/c_link.h>

languaget *new_clang_c_language()
{
  return new clang_c_languaget;
}

clang_c_languaget::clang_c_languaget()
{
  // Create a temporary directory, to dump clang's headers
  auto p = boost::filesystem::temp_directory_path();
  if(!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
  {
    std::cerr << "Can't find temporary directory (needed to dump clang headers)"
              << std::endl;
    abort();
  }

  // Build the compile arguments
  build_compiler_args(std::move(p.string()));

  // Dump clang headers on the temporary folder
  dump_clang_headers(p.string());
}

void clang_c_languaget::build_compiler_args(const std::string &&tmp_dir)
{
  compiler_args.emplace_back("clang-tool");

  compiler_args.push_back("-I" + tmp_dir);

  // Append mode arg
  switch(config.ansi_c.word_size)
  {
  case 16:
    compiler_args.emplace_back("-m16");
    break;

  case 32:
    compiler_args.emplace_back("-m32");
    break;

  case 64:
    compiler_args.emplace_back("-m64");
    break;

  default:
    std::cerr << "Unknown word size: " << config.ansi_c.word_size << std::endl;
    abort();
  }

  if(config.ansi_c.char_is_unsigned)
    compiler_args.emplace_back("-funsigned-char");

  if(config.options.get_bool_option("deadlock-check"))
  {
    compiler_args.emplace_back("-Dpthread_join=pthread_join_switch");
    compiler_args.emplace_back("-Dpthread_mutex_lock=pthread_mutex_lock_check");
    compiler_args.emplace_back(
      "-Dpthread_mutex_unlock=pthread_mutex_unlock_check");
    compiler_args.emplace_back("-Dpthread_cond_wait=pthread_cond_wait_check");
  }
  else if(config.options.get_bool_option("lock-order-check"))
  {
    compiler_args.emplace_back("-Dpthread_join=pthread_join_noswitch");
    compiler_args.emplace_back(
      "-Dpthread_mutex_lock=pthread_mutex_lock_nocheck");
    compiler_args.emplace_back(
      "-Dpthread_mutex_unlock=pthread_mutex_unlock_nocheck");
    compiler_args.emplace_back("-Dpthread_cond_wait=pthread_cond_wait_nocheck");
  }
  else
  {
    compiler_args.emplace_back("-Dpthread_join=pthread_join_noswitch");
    compiler_args.emplace_back(
      "-Dpthread_mutex_lock=pthread_mutex_lock_noassert");
    compiler_args.emplace_back(
      "-Dpthread_mutex_unlock=pthread_mutex_unlock_noassert");
    compiler_args.emplace_back("-Dpthread_cond_wait=pthread_cond_wait_nocheck");
  }

  for(auto const &def : config.ansi_c.defines)
    compiler_args.push_back("-D" + def);

  for(auto const &inc : config.ansi_c.include_paths)
    compiler_args.push_back("-I" + inc);

  // Ignore ctype defined by the system
  compiler_args.emplace_back("-D__NO_CTYPE");

#ifdef __APPLE__
  compiler_args.push_back("-D_EXTERNALIZE_CTYPE_INLINES_");
  compiler_args.push_back("-D_SECURE__STRING_H_");
  compiler_args.push_back("-U__BLOCKS__");
#endif

  // Force clang see all files as .c
  // This forces the preprocessor to be called even in preprocessed files
  // which allow us to perform transformations using -D
  compiler_args.emplace_back("-x");
  compiler_args.emplace_back("c");

  // Add -Wunknown-attributes, preprocessed files with GCC generate a bunch
  // of __leaf__ attributes that we don't care about
  compiler_args.emplace_back("-Wno-unknown-attributes");

  // Option to avoid creating a linking command
  compiler_args.emplace_back("-fsyntax-only");
}

bool clang_c_languaget::parse(
  const std::string &path,
  message_handlert &message_handler)
{
  // preprocessing

  std::ostringstream o_preprocessed;
  if(preprocess(path, o_preprocessed, message_handler))
    return true;

  // Get compiler arguments and add the file path
  std::vector<std::string> new_compiler_args(compiler_args);
  new_compiler_args.push_back(path);

  // Get intrinsics
  std::string intrinsics = internal_additions();

  // Generate ASTUnit and add to our vector
  auto AST = buildASTs(intrinsics, new_compiler_args);

  ASTs.push_back(std::move(AST));

  // Use diagnostics to find errors, rather than the return code.
  for(auto const &astunit : ASTs)
    if(astunit->getDiagnostics().hasErrorOccurred())
      return true;

  return false;
}

bool clang_c_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  contextt new_context;

  clang_c_convertert converter(new_context, ASTs);
  if(converter.convert())
    return true;

  clang_c_adjust adjuster(new_context);
  if(adjuster.adjust())
    return true;

  if(c_link(context, new_context, message_handler, module))
    return true;

  // Remove unused
  if(!config.options.get_bool_option("keep-unused"))
    context.remove_unused();

  return false;
}

void clang_c_languaget::show_parse(std::ostream &out __attribute__((unused)))
{
  for(auto const &translation_unit : ASTs)
    (*translation_unit).getASTContext().getTranslationUnitDecl()->dump();
}

bool clang_c_languaget::preprocess(
  const std::string &path __attribute__((unused)),
  std::ostream &outstream __attribute__((unused)),
  message_handlert &message_handler __attribute__((unused)))
{
// TODO: Check the preprocess situation.
#if 0
  return c_preprocess(path, outstream, false, message_handler);
#endif
  return false;
}

bool clang_c_languaget::final(
  contextt &context,
  message_handlert &message_handler)
{
  add_cprover_library(context, message_handler);
  return clang_main(context, "main", message_handler);
}

std::string clang_c_languaget::internal_additions()
{
  std::string intrinsics =
    R"(
# 1 "esbmc_intrinsics.h" 1
void __ESBMC_assume(_Bool);
void assert(_Bool);
void __ESBMC_assert(_Bool, const char *);
_Bool __ESBMC_same_object(const void *, const void *);
void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();

int __ESBMC_abs(int);
long int __ESBMC_labs(long int);
long long int __ESBMC_llabs(long long int);

// pointers
unsigned __ESBMC_POINTER_OBJECT(const void *);
signed __ESBMC_POINTER_OFFSET(const void *);

// malloc
__attribute__((used))
__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_alloc[1];

__attribute__((used))
__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_deallocated[1];

__attribute__((used))
__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_is_dynamic[1];

__attribute__((used))
__attribute__((annotate("__ESBMC_inf_size")))
unsigned long __ESBMC_alloc_size[1];

// float stuff
int __ESBMC_rounding_mode = 0;
_Bool __ESBMC_floatbv_mode();

double __ESBMC_fabsd(double);
long double __ESBMC_fabsld(long double);
float __ESBMC_fabsf(float);

_Bool __ESBMC_isnanf(float);
_Bool __ESBMC_isnand(double);
_Bool __ESBMC_isnanld(long double);

_Bool __ESBMC_isfinite(double);
_Bool __ESBMC_isfinitef(float);
_Bool __ESBMC_isfinited(double);
_Bool __ESBMC_isfiniteld(long double);

_Bool __ESBMC_isinf(double);
_Bool __ESBMC_isinff(float);
_Bool __ESBMC_isinfd(double);
_Bool __ESBMC_isinfld(long double);

_Bool __ESBMC_sign(double);
_Bool __ESBMC_signf(float);
_Bool __ESBMC_signd(double);
_Bool __ESBMC_signld(long double);

_Bool __ESBMC_isnormal(double);
_Bool __ESBMC_isnormalf(float);
_Bool __ESBMC_isnormald(double);
_Bool __ESBMC_isnormalld(long double);

double __ESBMC_inf(void);
float __ESBMC_inff(void);
long double __ESBMC_infld(void);

float __ESBMC_nearbyintf(float);
double __ESBMC_nearbyintd(double);
long double __ESBMC_nearbyintld(long double);

float __ESBMC_fmaf(float, float, float);
double __ESBMC_fmad(double, double, double);
long double __ESBMC_fmald(long double, long double, long double);

float __ESBMC_sqrtf(float);
double __ESBMC_sqrtd(double);
long double __ESBMC_sqrtld(long double);

void *__ESBMC_memset(void *, int, unsigned int);

// Digital controllers code
void __ESBMC_generate_cascade_controllers(float *, int, float *, int, _Bool);
void __ESBMC_generate_delta_coefficients(float *, double *, float);
_Bool __ESBMC_check_delta_stability(double *, double, int, int);

// Forward decs for pthread main thread begin/end hooks. Because they're
// pulled in from the C library, they need to be declared prior to pulling
// them in, for type checking.
__attribute__((used)) void pthread_start_main_hook(void);
__attribute__((used)) void pthread_end_main_hook(void);

// Forward declarations for nondeterministic types.
int nondet_int();
unsigned int nondet_uint();
long nondet_long();
unsigned long nondet_ulong();
short nondet_short();
unsigned short nondet_ushort();
char nondet_char();
unsigned char nondet_uchar();
signed char nondet_schar();
_Bool nondet_bool();
float nondet_float();
double nondet_double();

// TACAS definitions,
int __VERIFIER_nondet_int();
unsigned int __VERIFIER_nondet_uint();
long __VERIFIER_nondet_long();
unsigned long __VERIFIER_nondet_ulong();
short __VERIFIER_nondet_short();
unsigned short __VERIFIER_nondet_ushort();
char __VERIFIER_nondet_char();
unsigned char __VERIFIER_nondet_uchar();
signed char __VERIFIER_nondet_schar();
_Bool __VERIFIER_nondet_bool();
float __VERIFIER_nondet_float();
double __VERIFIER_nondet_double();

void __VERIFIER_error();
void __VERIFIER_assume(int);
void __VERIFIER_atomic_begin();
void __VERIFIER_atomic_end();
    )";

  return intrinsics;
}

bool clang_c_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns,
  bool fullname)
{
  code = expr2c(expr, ns, fullname);
  return false;
}

bool clang_c_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns,
  bool fullname)
{
  code = type2c(type, ns, fullname);
  return false;
}
