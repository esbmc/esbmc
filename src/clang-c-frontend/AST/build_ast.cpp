/*
 * build_ast.cpp
 *
 *  Created on: Apr 14, 2017
 *      Author: mramalho
 */

#include "build_ast.h"

#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/Tooling.h>

std::unique_ptr<clang::ASTUnit> buildASTs(
  std::string intrinsics,
  std::vector<std::string> compiler_args,
  std::vector<std::string> clang_headers_name,
  std::vector<std::string> clang_headers_content)
{
  // Create virtual file system to add clang's headers
  llvm::IntrusiveRefCntPtr<clang::vfs::OverlayFileSystem> OverlayFileSystem(
    new clang::vfs::OverlayFileSystem(clang::vfs::getRealFileSystem()));

  llvm::IntrusiveRefCntPtr<clang::vfs::InMemoryFileSystem> InMemoryFileSystem(
    new clang::vfs::InMemoryFileSystem);

  OverlayFileSystem->pushOverlay(InMemoryFileSystem);

  llvm::IntrusiveRefCntPtr<clang::FileManager> Files(
    new clang::FileManager(clang::FileSystemOptions(), OverlayFileSystem));

  clang::tooling::ToolInvocation Invocation(
    compiler_args, new clang::ASTPrintAction, Files.get());

  Invocation.run();

  return nullptr;
}
