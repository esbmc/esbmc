/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <llvm_language.h>
#include <llvm_typecheck.h>

static llvm::cl::OptionCategory esbmc_llvm("esmc_llvm");

llvm_languaget::llvm_languaget(std::vector<std::string> _files)
  : files(_files)
{
  // From the clang tool example,
  int num_args = 2 + _files.size();
  const char **the_args = (const char**) malloc(sizeof(const char*) * num_args);

  int i=0;
  the_args[i++] = "clang";
  for(; i <= _files.size(); ++i)
    the_args[i] = _files.at(i-1).c_str();
  the_args[i] = "--";

  OptionsParser = new clang::tooling::CommonOptionsParser(num_args, the_args, esbmc_llvm);
  free(the_args);

  Tool = new clang::tooling::ClangTool(OptionsParser->getCompilations(),
    OptionsParser->getSourcePathList());
}

llvm_languaget::~llvm_languaget()
{
  delete OptionsParser;
  delete Tool;
}

bool llvm_languaget::parse()
{
  Tool->buildASTs(ASTs);

  // Use diagnostics to find errors, rather than the return code.
  for (const auto &astunit : ASTs) {
    if (astunit->getDiagnostics().hasErrorOccurred()) {
      std::cerr << std::endl;
      return true;
    }
  }

  return false;
}

bool llvm_languaget::preprocess(
  const std::string &path,
  std::ostream &outstream,
  message_handlert &message_handler)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

void llvm_languaget::internal_additions(std::ostream &out)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
}

bool llvm_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  contextt new_context;
  llvm_typecheckt typecheck(context);
  typecheck.ASTs.swap(ASTs);

  return typecheck.typecheck();
}

bool llvm_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

bool llvm_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}

bool llvm_languaget::to_expr(
  const std::string &code,
  const std::string &module __attribute__((unused)),
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns)
{
  std::cout << "Method " << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
  abort();
  return true;
}
