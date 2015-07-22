/*
 * llvmparsert.cpp
 *
 *  Created on: Jul 22, 2015
 *      Author: mramalho
 */

#include <iostream>
#include <string>
#include <cstdlib>

#include "expr.h"

#include "llvm_parser.h"

#include <clang/AST/ASTConsumer.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static llvm::cl::OptionCategory esbmc_llvm("esmc_llvm");
llvm_parsert llvm_parser;

llvm_parsert::llvm_parsert()
{
}

llvm_parsert::~llvm_parsert()
{

}

bool llvm_parsert::parse()
{
  // From the clang tool example,
  int num_args = 3;
  const char **the_args = (const char**) malloc(sizeof(const char*) * num_args);

  unsigned int i = 0;
  the_args[i++] = "clang";
  the_args[i++] = filename.c_str();
  the_args[i++] = "--";

  CommonOptionsParser OptionsParser(num_args, the_args, esbmc_llvm);
  free(the_args);

  ClangTool Tool(OptionsParser.getCompilations(),
    OptionsParser.getSourcePathList());

  Tool.buildASTs(ASTs);

  return false;
}
