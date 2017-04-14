/*
 * build_ast.cpp
 *
 *  Created on: Apr 14, 2017
 *      Author: mramalho
 */

#include "build_ast.h"

#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/Tooling.h>

class ASTBuilderAction : public clang::tooling::ToolAction
{
  std::vector<std::unique_ptr<clang::ASTUnit>> &ASTs;

public:
  ASTBuilderAction(std::vector<std::unique_ptr<clang::ASTUnit>> &ASTs)
    : ASTs(ASTs)
  {}

  bool runInvocation(
    std::shared_ptr<clang::CompilerInvocation> Invocation,
    clang::FileManager *Files,
    std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps,
    clang::DiagnosticConsumer *DiagConsumer) override
  {
    std::unique_ptr<clang::ASTUnit> AST =
      clang::ASTUnit::LoadFromCompilerInvocation(
        Invocation, std::move(PCHContainerOps),
        clang::CompilerInstance::createDiagnostics(&Invocation->getDiagnosticOpts(),
          DiagConsumer,
          /*ShouldOwnClient=*/false),
        Files);

    if (!AST)
      return false;

    ASTs.push_back(std::move(AST));
    return true;
  }
};

int buildASTs(
  clang::tooling::ClangTool tool,
  std::vector<std::unique_ptr<clang::ASTUnit>> &ASTs)
{
  ASTBuilderAction Action(ASTs);
  return tool.run(&Action);
}
