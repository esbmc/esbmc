/*
 * esbmcaction.h
 *
 *  Created on: Apr 25, 2017
 *      Author: mramalho
 */

#ifndef CLANG_C_FRONTEND_AST_ESBMC_ACTION_H_
#define CLANG_C_FRONTEND_AST_ESBMC_ACTION_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Lex/Preprocessor.h>
#pragma GCC diagnostic pop
#include <string>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

class esbmc_action : public clang::ASTFrontendAction
{
public:
  esbmc_action(const std::string &&esbmc_instrinsics)
    : intrinsics(esbmc_instrinsics){};

  bool BeginSourceFileAction(clang::CompilerInstance &CI) override
  {
    clang::Preprocessor &PP = CI.getPreprocessor();

    std::string s = PP.getPredefines();
    s += intrinsics;
    PP.setPredefines(s);

    return true;
  }

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override
  {
    return llvm::make_unique<clang::ASTConsumer>();
  }

  std::string intrinsics;
};

#endif /* CLANG_C_FRONTEND_AST_ESBMC_ACTION_H_ */
