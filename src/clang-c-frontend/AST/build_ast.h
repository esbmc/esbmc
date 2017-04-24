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
  class ASTUnit;
}

std::unique_ptr<clang::ASTUnit> buildASTs(
  std::string path,
  std::string intrinsics,
  std::vector<std::string> compiler_args,
  std::vector<std::string> clang_headers_name,
  std::vector<std::string> clang_headers_content);

#endif /* CLANG_C_FRONTEND_AST_BUILD_AST_H_ */
