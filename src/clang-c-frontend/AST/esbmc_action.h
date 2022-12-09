#include <util/compiler_defs.h>
#ifndef CLANG_C_FRONTEND_AST_ESBMC_ACTION_H_
#define CLANG_C_FRONTEND_AST_ESBMC_ACTION_H_

#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Lex/Preprocessor.h>
CC_DIAGNOSTIC_POP()
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
    return std::make_unique<clang::ASTConsumer>();
  }

  std::string intrinsics;
};

#endif /* CLANG_C_FRONTEND_AST_ESBMC_ACTION_H_ */
