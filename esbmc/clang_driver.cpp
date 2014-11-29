#include "parseoptions.h"

#include <clang/AST/ASTConsumer.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static llvm::cl::OptionCategory dummy_tool_cat("dummy-tool-cat");

std::vector<std::unique_ptr<clang::ASTUnit> >
cbmc_parseoptionst::parse_clang()
{

  // From the clang tool example,
  int num_args = cmdline.args.size();
  num_args += 6;
  const char **the_args = malloc(sizeof(const char*) * num_args);
  the_args[0] = "clang";
  unsigned int i = 1;
  for (const std::string &str : cmdline.args)
    the_args[i++] = str.c_str();
  the_args[i++] = "--";
  the_args[i++] = "-I";
  the_args[i++] = "/home/jmorse/phd/esbmc/ansi-c/headers";
  the_args[i++] = "-D";
  the_args[i++] = "__ESBMC_CLANG_PARSER";

  CommonOptionsParser OptionsParser(num_args, the_args, dummy_tool_cat);
  free(the_args);

  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  std::vector<std::unique_ptr<clang::ASTUnit> > ASTs;

  Tool.buildASTs(ASTs);

  return ASTs;
}
