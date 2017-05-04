/*
 * build_ast.h
 *
 *  Created on: Apr 14, 2017
 *      Author: mramalho
 */

#ifndef CLANG_C_FRONTEND_AST_BUILD_AST_H_
#define CLANG_C_FRONTEND_AST_BUILD_AST_H_

#include <memory>
#include <unordered_map>
#include <vector>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

// Forward dec, to avoid bringing in clang headers
namespace clang {
  class ASTUnit;
}

std::unique_ptr<clang::ASTUnit> buildASTs(
  std::string intrinsics,
  std::vector<std::string> compiler_args);

#endif /* CLANG_C_FRONTEND_AST_BUILD_AST_H_ */
