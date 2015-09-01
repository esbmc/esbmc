/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <sstream>
#include <fstream>

#include <llvm_language.h>
#include <llvm_convert.h>
#include <llvm_adjust.h>
#include <llvm_main.h>

#include <ansi-c/cprover_library.h>
#include <ansi-c/c_preprocess.h>
#include <ansi-c/c_link.h>
#include <ansi-c/gcc_builtin_headers.h>

#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

static llvm::cl::OptionCategory esbmc_llvm("esmc_llvm");

languaget *new_llvm_language()
{
  return new llvm_languaget;
}

llvm_languaget::llvm_languaget()
{
}

llvm_languaget::~llvm_languaget()
{
}

bool llvm_languaget::parse(
  const std::string &path,
  message_handlert &message_handler)
{
  // preprocessing

  std::ostringstream o_preprocessed;
  if(preprocess(path, o_preprocessed, message_handler))
    return true;

  return parse(path);
}

bool llvm_languaget::parse(const std::string& path)
{
  // From the clang tool example,
  int num_args = 3;
  const char **the_args = (const char**) malloc(sizeof(const char*) * num_args);

  unsigned int i=0;
  the_args[i++] = "clang";
  the_args[i++] = path.c_str();
  the_args[i] = "--";

  clang::tooling::CommonOptionsParser OptionsParser(
    num_args,
    the_args,
    esbmc_llvm);
  free(the_args);

  clang::tooling::ClangTool Tool(
    OptionsParser.getCompilations(),
    OptionsParser.getSourcePathList());

  Tool.buildASTs(ASTs);

  // Use diagnostics to find errors, rather than the return code.
  for (const auto &astunit : ASTs) {
    if (astunit->getDiagnostics().hasErrorOccurred()) {
      std::cerr << std::endl;
      return true;
    }
  }

  return false;
}

bool llvm_languaget::typecheck(
  contextt& context,
  const std::string& module,
  message_handlert& message_handler)
{
  return convert(context, module, message_handler);
}

void llvm_languaget::show_parse(std::ostream& out __attribute__((unused)))
{
  for (auto &translation_unit : ASTs)
  {
    for (clang::ASTUnit::top_level_iterator
      it = translation_unit->top_level_begin();
      it != translation_unit->top_level_end();
      it++)
    {
      (*it)->dump();
    }
  }
}

bool llvm_languaget::convert(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  contextt new_context;

  llvm_convertert converter(new_context, ASTs);
  if(converter.convert())
    return true;

  llvm_adjust adjuster(new_context);
  if(adjuster.adjust())
    return true;

  return c_link(context, new_context, message_handler, module);
}

bool llvm_languaget::preprocess(
  const std::string &path,
  std::ostream &outstream,
  message_handlert &message_handler)
{
  if(path=="")
    return c_preprocess("", outstream, true, message_handler);

  // check extension

  const char *ext=strrchr(path.c_str(), '.');
  if(ext!=NULL && std::string(ext)==".ipp")
  {
    std::ifstream infile(path.c_str());

    char ch;

    while(infile.read(&ch, 1))
      outstream << ch;

    return false;
  }

  return c_preprocess(path, outstream, true, message_handler);
}

bool llvm_languaget::final(contextt& context, message_handlert& message_handler)
{
  add_cprover_library(context, message_handler);
  if(llvm_main(context, "c::", "c::main", message_handler)) return true;
  return false;
}

void llvm_languaget::internal_additions(std::string &str)
{
  str+=
    "void __ESBMC_assume(_Bool assumption);\n"
    "void assert(_Bool assertion);\n"
    "void __ESBMC_assert(_Bool assertion, const char *description);\n"
    "_Bool __ESBMC_same_object(const void *, const void *);\n"
    "_Bool __ESBMC_is_zero_string(const void *);\n"
    "unsigned __ESBMC_zero_string_length(const void *);\n"
    "unsigned __ESBMC_buffer_size(const void *);\n"

    // traces
    "void CBMC_trace(int lvl, const char* event, ...);\n"

    // pointers
    "unsigned __ESBMC_POINTER_OBJECT(const void *p);\n"
    "signed __ESBMC_POINTER_OFFSET(const void *p);\n"

    // malloc
    "unsigned __ESBMC_constant_infinity_uint;\n"
    //"_Bool __ESBMC_alloc[__ESBMC_constant_infinity_uint];\n"
    //"_Bool __ESBMC_deallocated[__ESBMC_constant_infinity_uint];\n"
    //"_Bool __ESBMC_is_dynamic[__ESBMC_constant_infinity_uint];\n"
    //"unsigned long __ESBMC_alloc_size[__ESBMC_constant_infinity_uint];\n"

    "void *__ESBMC_realloc(void *ptr, long unsigned int size);\n"

    // float stuff
    "_Bool __ESBMC_isnan(double f);\n"
    "_Bool __ESBMC_isfinite(double f);\n"
    "_Bool __ESBMC_isinf(double f);\n"
    "_Bool __ESBMC_isnormal(double f);\n"
    "extern int __ESBMC_rounding_mode;\n"

    // absolute value
    "int __ESBMC_abs(int x);\n"
    "long int __ESBMC_labs(long int x);\n"
    "double __ESBMC_fabs(double x);\n"
    "long double __ESBMC_fabsl(long double x);\n"
    "float __ESBMC_fabsf(float x);\n"

    // Forward decs for pthread main thread begin/end hooks. Because they're
    // pulled in from the C library, they need to be declared prior to pulling
    // them in, for type checking.
    "void pthread_start_main_hook(void);\n"
    "void pthread_end_main_hook(void);\n"

    // Forward declarations for nondeterministic types.
    "int nondet_int();\n"
    "unsigned int nondet_uint();\n"
    "long nondet_long();\n"
    "unsigned long nondet_ulong();\n"
    "short nondet_short();\n"
    "unsigned short nondet_ushort();\n"
    "short nondet_short();\n"
    "unsigned short nondet_ushort();\n"
    "char nondet_char();\n"
    "unsigned char nondet_uchar();\n"
    "signed char nondet_schar();\n"

    // Digital filters code
    "_Bool __ESBMC_check_stability(float den[], float num[]);\n"

    // Digital controllers code
    "void __ESBMC_generate_cascade_controllers(float * cden, int csize, float * cout, int coutsize, _Bool isDenominator);\n"
    "void __ESBMC_generate_delta_coefficients(float a[], double out[], float delta);\n"
    "_Bool __ESBMC_check_delta_stability(double dc[], double sample_time, int iwidth, int precision);\n"

    // And again, for TACAS VERIFIER versions,
    "int __VERIFIER_nondet_int();\n"
    "unsigned int __VERIFIER_nondet_uint();\n"
    "long __VERIFIER_nondet_long();\n"
    "unsigned long __VERIFIER_nondet_ulong();\n"
    "short __VERIFIER_nondet_short();\n"
    "unsigned short __VERIFIER_nondet_ushort();\n"
    "short __VERIFIER_nondet_short();\n"
    "unsigned short __VERIFIER_nondet_ushort();\n"
    "char __VERIFIER_nondet_char();\n"
    "unsigned char __VERIFIER_nondet_uchar();\n"
    "signed char __VERIFIER_nondet_schar();\n"

    // GCC junk stuff
//    GCC_BUILTIN_HEADERS

    "\n";
}

bool llvm_languaget::from_expr(
  const exprt &expr __attribute__((unused)),
  std::string &code __attribute__((unused)),
  const namespacet &ns __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

bool llvm_languaget::from_type(
  const typet &type __attribute__((unused)),
  std::string &code __attribute__((unused)),
  const namespacet &ns __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

bool llvm_languaget::to_expr(
  const std::string &code __attribute__((unused)),
  const std::string &module __attribute__((unused)),
  exprt &expr __attribute__((unused)),
  message_handlert &message_handler __attribute__((unused)),
  const namespacet &ns __attribute__((unused)))
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}
