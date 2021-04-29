/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <clang/Frontend/ASTUnit.h>
#pragma GCC diagnostic pop

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
#include <util/message/format.h>
#include <util/filesystem.h>

#include <ac_config.h>

languaget *new_clang_c_language(const messaget &msg)
{
  return new clang_c_languaget(msg);
}

clang_c_languaget::clang_c_languaget(const messaget &msg) : languaget(msg)
{
  // Build the compile arguments
  build_compiler_args(clang_headers_path());
}

void clang_c_languaget::build_compiler_args(const std::string &tmp_dir)
{
  compiler_args.emplace_back("clang-tool");

  const std::string *libc_headers = internal_libc_header_dir();
  if(libc_headers)
  {
    compiler_args.push_back("-isystem");
    compiler_args.push_back(*libc_headers);
  }

  compiler_args.push_back("-isystem");
  compiler_args.push_back(tmp_dir);

  // Append mode arg
  switch(config.ansi_c.word_size)
  {
  case 16:
  case 32:
  case 64:
    compiler_args.emplace_back("-m" + std::to_string(config.ansi_c.word_size));
    break;

  default:
    msg.error(
      fmt::format("Unknown word size: {}\n", config.ansi_c.word_size).c_str());
    abort();
  }

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
    compiler_args.emplace_back(
      "-Dpthread_mutex_destroy=pthread_mutex_destroy_check");
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

  if(msg.get_verbosity() >= VerbosityLevel::Debug)
    compiler_args.emplace_back("-v");

  compiler_args.emplace_back("-target");
  compiler_args.emplace_back(config.ansi_c.target.to_string());

  std::string sysroot = config.options.get_option("sysroot");
  if(!sysroot.empty())
    compiler_args.push_back("--sysroot=" + sysroot);

  for(const auto &dir : config.ansi_c.idirafter_paths)
  {
    compiler_args.push_back("-idirafter");
    compiler_args.push_back(dir);
  }

  for(auto const &inc : config.ansi_c.include_paths)
    compiler_args.push_back("-I" + inc);

  for(auto const &inc : config.ansi_c.forces)
    compiler_args.push_back("-f" + inc);

  for(auto const &inc : config.ansi_c.warnings)
    compiler_args.push_back("-W" + inc);

  compiler_args.emplace_back("-D__builtin_sadd_overflow=__ESBMC_overflow_sadd");
  compiler_args.emplace_back(
    "-D__builtin_saddl_overflow=__ESBMC_overflow_saddl");
  compiler_args.emplace_back(
    "-D__builtin_saddll_overflow=__ESBMC_overflow_saddll");
  compiler_args.emplace_back("-D__builtin_uadd_overflow=__ESBMC_overflow_uadd");
  compiler_args.emplace_back(
    "-D__builtin_uaddl_overflow=__ESBMC_overflow_uaddl");
  compiler_args.emplace_back(
    "-D__builtin_uaddll_overflow=__ESBMC_overflow_uaddll");

  compiler_args.emplace_back("-D__builtin_ssub_overflow=__ESBMC_overflow_ssub");
  compiler_args.emplace_back(
    "-D__builtin_ssubl_overflow=__ESBMC_overflow_ssubl");
  compiler_args.emplace_back(
    "-D__builtin_ssubll_overflow=__ESBMC_overflow_ssubll");
  compiler_args.emplace_back("-D__builtin_usub_overflow=__ESBMC_overflow_usub");
  compiler_args.emplace_back(
    "-D__builtin_usubl_overflow=__ESBMC_overflow_usubl");
  compiler_args.emplace_back(
    "-D__builtin_usubll_overflow=__ESBMC_overflow_usubll");

  compiler_args.emplace_back("-D__builtin_smul_overflow=__ESBMC_overflow_smul");
  compiler_args.emplace_back(
    "-D__builtin_smull_overflow=__ESBMC_overflow_smull");
  compiler_args.emplace_back(
    "-D__builtin_smulll_overflow=__ESBMC_overflow_smulll");
  compiler_args.emplace_back("-D__builtin_umul_overflow=__ESBMC_overflow_umul");
  compiler_args.emplace_back(
    "-D__builtin_umull_overflow=__ESBMC_overflow_umull");
  compiler_args.emplace_back(
    "-D__builtin_umulll_overflow=__ESBMC_overflow_umulll");
  compiler_args.emplace_back(
    "-D__sync_fetch_and_add=__ESBMC_sync_fetch_and_add");
  compiler_args.emplace_back(
    "-D__builtin_constant_p=__ESBMC_builtin_constant_p");

  compiler_args.emplace_back("-D__builtin_memcpy=memcpy");

  compiler_args.emplace_back("-D__ESBMC_alloca=__builtin_alloca");

  // Ignore ctype defined by the system
  compiler_args.emplace_back("-D__NO_CTYPE");

  if(config.ansi_c.target.is_macos())
  {
    compiler_args.push_back("-D_EXTERNALIZE_CTYPE_INLINES_");
    compiler_args.push_back("-D_SECURE__STRING_H_");
    compiler_args.push_back("-U__BLOCKS__");
    compiler_args.push_back("-Wno-nullability-completeness");
    compiler_args.push_back("-Wno-deprecated-register");
  }

  if(config.ansi_c.target.is_windows_abi())
  {
    compiler_args.push_back("-D_INC_TIME_INL");
    compiler_args.push_back("-D__CRT__NO_INLINE");
    compiler_args.push_back("-D_USE_MATH_DEFINES");
  }

  // Increase maximum bracket depth
  compiler_args.push_back("-fbracket-depth=1024");

  // Add -Wunknown-attributes, preprocessed files with GCC generate a bunch
  // of __leaf__ attributes that we don't care about
  compiler_args.emplace_back("-Wno-unknown-attributes");

  // Option to avoid creating a linking command
  compiler_args.emplace_back("-fsyntax-only");
}

void clang_c_languaget::force_file_type()
{
  // Force clang see all files as .c
  // This forces the preprocessor to be called even in preprocessed files
  // which allow us to perform transformations using -D
  compiler_args.push_back("-x");
  compiler_args.push_back("c");
}

bool clang_c_languaget::parse(const std::string &path, const messaget &msg)
{
  // preprocessing

  std::ostringstream o_preprocessed;
  if(preprocess(path, o_preprocessed, msg))
    return true;

  // Force the file type, .c for the C frontend and .cpp for the C++ one
  force_file_type();

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
  const messaget &msg)
{
  contextt new_context(msg);

  clang_c_convertert converter(new_context, ASTs, msg);
  if(converter.convert())
    return true;

  clang_c_adjust adjuster(new_context, msg);
  if(adjuster.adjust())
    return true;

  if(c_link(context, new_context, msg, module))
    return true;

  return false;
}

void clang_c_languaget::show_parse(std::ostream &)
{
  for(auto const &translation_unit : ASTs)
    (*translation_unit).getASTContext().getTranslationUnitDecl()->dump();
}

bool clang_c_languaget::preprocess(
  const std::string &,
  std::ostream &,
  const messaget &)
{
// TODO: Check the preprocess situation.
#if 0
  return c_preprocess(path, outstream, false, message_handler);
#endif
  return false;
}

bool clang_c_languaget::final(contextt &context, const messaget &msg)
{
  add_cprover_library(context, msg, this);
  return clang_main(context, msg);
}

std::string clang_c_languaget::internal_additions()
{
  std::string intrinsics =
    R"(
# 1 "esbmc_intrinsics.h" 1
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
_Bool __ESBMC_same_object(const void *, const void *);
void __ESBMC_yield();
void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();
void __ESBMC_init_var(void*);

int __ESBMC_abs(int);
long int __ESBMC_labs(long int);
long long int __ESBMC_llabs(long long int);

// pointers
unsigned __ESBMC_POINTER_OBJECT(const void *);
signed __ESBMC_POINTER_OFFSET(const void *);

// malloc
__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_alloc[1];

__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_deallocated[1];

__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_is_dynamic[1];

__attribute__((annotate("__ESBMC_inf_size")))
unsigned __ESBMC_alloc_size[1];

// Get object size
unsigned __ESBMC_get_object_size(const void *);


_Bool __ESBMC_is_little_endian();

int __ESBMC_rounding_mode = 0;

void *__ESBMC_memset(void *, int, unsigned int);

// Forward decs for pthread main thread begin/end hooks. Because they're
// pulled in from the C library, they need to be declared prior to pulling
// them in, for type checking.
void pthread_start_main_hook(void);
void pthread_end_main_hook(void);

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

_Bool __ESBMC_overflow_sadd(int, int, int *);
_Bool __ESBMC_overflow_saddl(long int, long int, long int *);
_Bool __ESBMC_overflow_saddll(long long int, long long int, long long int *);
_Bool __ESBMC_overflow_uadd(unsigned int, unsigned int, unsigned int *);
_Bool __ESBMC_overflow_uaddl(unsigned long int, unsigned long int, unsigned long int *);
_Bool __ESBMC_overflow_uaddll(unsigned long long int, unsigned long long int, unsigned long long int *);
_Bool __ESBMC_overflow_ssub(int, int, int *);
_Bool __ESBMC_overflow_ssubl(long int, long int, long int *);
_Bool __ESBMC_overflow_ssubll(long long int, long long int, long long int *);
_Bool __ESBMC_overflow_usub(unsigned int, unsigned int, unsigned int *);
_Bool __ESBMC_overflow_usubl(unsigned long int, unsigned long int, unsigned long int *);
_Bool __ESBMC_overflow_usubll(unsigned long long int, unsigned long long int, unsigned long long int *);
_Bool __ESBMC_overflow_smul(int, int, int *);
_Bool __ESBMC_overflow_smull(long int, long int, long int *);
_Bool __ESBMC_overflow_smulll(long long int, long long int, long long int *);
_Bool __ESBMC_overflow_umul(unsigned int, unsigned int, unsigned int *);
_Bool __ESBMC_overflow_umull(unsigned long int, unsigned long int, unsigned long int *);
_Bool __ESBMC_overflow_umulll(unsigned long long int, unsigned long long int, unsigned long long int *);
int __ESBMC_sync_fetch_and_add(int*, int);
int __ESBMC_builtin_constant_p(int);

// This is causing problems when using the C++ frontend. It needs to be rewritten
#define __atomic_load_n(PTR, MO)                                               \
  __extension__({                                                              \
    __auto_type __atomic_load_ptr = (PTR);                                     \
    __typeof__(*__atomic_load_ptr) __atomic_load_tmp;                          \
    __ESBMC_atomic_load(__atomic_load_ptr, &__atomic_load_tmp, (MO));          \
    __atomic_load_tmp;                                                         \
  })

#define __atomic_store_n(PTR, VAL, MO)                                         \
  __extension__({                                                              \
    __auto_type __atomic_store_ptr = (PTR);                                    \
    __typeof__(*__atomic_store_ptr) __atomic_store_tmp = (VAL);                \
    __ESBMC_atomic_store(__atomic_store_ptr, &__atomic_store_tmp, (MO));       \
  })
    )";

  return intrinsics;
}

bool clang_c_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  code = expr2c(expr, ns);
  return false;
}

bool clang_c_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  code = type2c(type, ns);
  return false;
}
