/*
 * build_ast.h
 *
 *  Created on: Apr 14, 2017
 *      Author: mramalho
 */

#ifndef CLANG_C_FRONTEND_AST_BUILD_AST_H_
#define CLANG_C_FRONTEND_AST_BUILD_AST_H_

#include <vector>
#include <memory>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

// Forward dec, to avoid bringing in clang headers
namespace clang {
  namespace tooling {
    class ClangTool;
  }

  class ASTUnit;
}

int buildASTs(
  clang::tooling::ClangTool tool,
  std::vector<std::unique_ptr<clang::ASTUnit>> &ASTs);

#endif /* CLANG_C_FRONTEND_AST_BUILD_AST_H_ */
