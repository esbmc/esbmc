/*
 * build_ast.cpp
 *
 *  Created on: Apr 14, 2017
 *      Author: mramalho
 */

#include "build_ast.h"

#include <clang/Frontend/ASTUnit.h>

std::unique_ptr<clang::ASTUnit> buildASTs(
  std::string path,
  std::string intrinsics,
  std::vector<std::string> compiler_args,
  std::vector<std::string> clang_headers_name,
  std::vector<std::string> clang_headers_content)
{
  return nullptr;
}
