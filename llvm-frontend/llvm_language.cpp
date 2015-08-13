/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <sstream>
#include <fstream>

#include <llvm_language.h>
#include <llvm_convert.h>
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

  internal_additions(o_preprocessed);

  if(preprocess(path, o_preprocessed, message_handler))
    return true;

  std::istringstream i_preprocessed(o_preprocessed.str());

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
  llvm_convertert converter(new_context);
  converter.ASTs.swap(ASTs);

  if(converter.convert())
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

void llvm_languaget::internal_additions(std::ostream &out)
{
  out << "# 1 \"<built-in>\"" << std::endl;

  out << "void *operator new(unsigned int size);" << std::endl;

  // assume/assert
  out << "extern \"C\" void assert(bool assertion);" << std::endl;
  out << "extern \"C\" void __ESBMC_assume(bool assumption);" << std::endl;
  out << "extern \"C\" void __ESBMC_assert("
         "bool assertion, const char *description);" << std::endl;

  // __ESBMC_atomic_{begin,end}
  out << "extern \"C\" void __ESBMC_atomic_begin();" << std::endl;
  out << "extern \"C\" void __ESBMC_atomic_end();" << std::endl;

  // __CPROVER namespace
  out << "namespace __CPROVER { }" << std::endl;

  // for dynamic objects
  out << "unsigned __CPROVER::constant_infinity_uint;" << std::endl;
  out << "bool __ESBMC_alloc[__CPROVER::constant_infinity_uint];" << std::endl;
  out << "unsigned __ESBMC_alloc_size[__CPROVER::constant_infinity_uint];" << std::endl;
  out << " bool __ESBMC_deallocated[__CPROVER::constant_infinity_uint];" << std::endl;
  out << "bool __ESBMC_is_dynamic[__CPROVER::constant_infinity_uint];" << std::endl;

  // GCC stuff
  out << "extern \"C\" {" << std::endl;
  out << GCC_BUILTIN_HEADERS;

  // Forward decs for pthread main thread begin/end hooks. Because they're
  // pulled in from the C library, they need to be declared prior to pulling
  // them in, for type checking.
  out << "void pthread_start_main_hook(void);" << std::endl;
  out << "void pthread_end_main_hook(void);" << std::endl;

  //  Empty __FILE__ and __LINE__ definitions.
  out << "const char *__FILE__ = \"\";" << std::endl;
  out << "unsigned int __LINE__ = 0;" << std::endl;

  out << "}" << std::endl;
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
