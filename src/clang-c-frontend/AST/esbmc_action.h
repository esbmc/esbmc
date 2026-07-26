#include <util/base/compiler_defs.h>
#ifndef CLANG_C_FRONTEND_AST_ESBMC_ACTION_H_
#  define CLANG_C_FRONTEND_AST_ESBMC_ACTION_H_

#  include <util/base/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#  include <clang/Frontend/CompilerInstance.h>
#  include <clang/Frontend/FrontendActions.h>
#  include <clang/Lex/Preprocessor.h>
CC_DIAGNOSTIC_POP()
#  include <string>

#  define __STDC_LIMIT_MACROS
#  define __STDC_FORMAT_MACROS

class esbmc_action : public clang::ASTFrontendAction
{
public:
  esbmc_action(const std::string &&esbmc_instrinsics)
    : intrinsics(esbmc_instrinsics){};

  bool BeginSourceFileAction(clang::CompilerInstance &CI) override
  {
    clang::Preprocessor &PP = CI.getPreprocessor();

    std::string s = PP.getPredefines();

    /* clang appends each -include / --include-file as an `#include` line at the
     * end of the predefines buffer. Such a header can transitively reach ESBMC's
     * own models, which use nondet_* / __ESBMC_*, so appending the intrinsics
     * after it leaves them undeclared at that point. Put them before the first
     * forced include instead -- still after the builtin #defines the intrinsics
     * rely on (github #5868). */
    const std::string first_include = "\n#include ";
    size_t pos = s.find(first_include);
    if (pos == std::string::npos)
      s += intrinsics;
    else
      s.insert(pos + 1, intrinsics + "\n");
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
