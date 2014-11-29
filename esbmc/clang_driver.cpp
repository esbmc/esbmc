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

int
cbmc_parseoptionst::parse_clang()
{

  // From the clang tool example,
  int num_args = cmdline.args.size();
  num_args += 2;
  const char **the_args = malloc(sizeof(const char*) * num_args);
  the_args[0] = "clang";
  unsigned int i = 1;
  for (const std::string &str : cmdline.args)
    the_args[i++] = str.c_str();
  the_args[i++] = "--";

  CommonOptionsParser OptionsParser(num_args, the_args, dummy_tool_cat);
  free(the_args);

  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  std::vector<std::unique_ptr<clang::ASTUnit> > ASTs;

  Tool.buildASTs(ASTs);

  abort();

  return 0;
}
