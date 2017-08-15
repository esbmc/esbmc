/*
 * build_ast.cpp
 *
 *  Created on: Apr 14, 2017
 *      Author: mramalho
 */

#include <clang-c-frontend/AST/build_ast.h>
#include <clang-c-frontend/AST/esbmc_action.h>
#include <clang/Basic/Version.inc>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Support/Path.h>

std::unique_ptr<clang::ASTUnit> buildASTs(
  const std::string &intrinsics,
  const std::vector<std::string> &compiler_args)
{
  // Create virtual file system to add clang's headers
  llvm::IntrusiveRefCntPtr<clang::vfs::OverlayFileSystem> OverlayFileSystem(
    new clang::vfs::OverlayFileSystem(clang::vfs::getRealFileSystem()));

  llvm::IntrusiveRefCntPtr<clang::vfs::InMemoryFileSystem> InMemoryFileSystem(
    new clang::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);

  llvm::IntrusiveRefCntPtr<clang::FileManager> Files(
    new clang::FileManager(clang::FileSystemOptions(), OverlayFileSystem));

  // Create everything needed to create a CompilerInvocation,
  // copied from ToolInvocation::run
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
    new clang::DiagnosticOptions();

  std::unique_ptr<llvm::opt::OptTable> Opts(clang::driver::createDriverOptTable());

  std::vector<const char*> Argv;
  for (const std::string &Str : compiler_args)
    Argv.push_back(Str.c_str());

  unsigned MissingArgIndex, MissingArgCount;
  llvm::opt::InputArgList ParsedArgs = Opts->ParseArgs(
    llvm::ArrayRef<const char *>(Argv).slice(1),
    MissingArgIndex,
    MissingArgCount);

  clang::ParseDiagnosticArgs(*DiagOpts, ParsedArgs);

  clang::TextDiagnosticPrinter *DiagnosticPrinter =
    new clang::TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  clang::DiagnosticsEngine *Diagnostics =
    new clang::DiagnosticsEngine(
      llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
        new clang::DiagnosticIDs()),
        &*DiagOpts,
        DiagnosticPrinter,
        false);

  const std::unique_ptr<clang::driver::Driver> Driver(
    new clang::driver::Driver(
      "clang-tool",
      llvm::sys::getDefaultTargetTriple(),
      *Diagnostics,
      std::move(Files->getVirtualFileSystem())));
  Driver->setTitle("clang_based_tool");

  // Since the input might only be virtual, don't check whether it exists.
  Driver->setCheckInputsExist(false);
  const std::unique_ptr<clang::driver::Compilation> Compilation(
    Driver->BuildCompilation(llvm::makeArrayRef(Argv)));

  const clang::driver::JobList &Jobs = Compilation->getJobs();
  assert(Jobs.size() == 1);

  const llvm::opt::ArgStringList *const CC1Args = &Jobs.begin()->getArguments();

#if (CLANG_VERSION_MAJOR >= 4)
  std::shared_ptr<clang::CompilerInvocation> Invocation(
    clang::tooling::newInvocation(Diagnostics, *CC1Args));
#else
  auto Invocation = clang::tooling::newInvocation(Diagnostics, *CC1Args);
#endif

  // Show the invocation, with -v.
  if (Invocation->getHeaderSearchOpts().Verbose) {
    llvm::errs() << "clang Invocation:\n";
    Compilation->getJobs().Print(llvm::errs(), "\n", true);
    llvm::errs() << "\n";
  }

  // Create our custom action
  auto action = new esbmc_action(std::move(intrinsics));

  // Create ASTUnit
  std::unique_ptr<clang::ASTUnit> unit(
    clang::ASTUnit::LoadFromCompilerInvocationAction(
#if (CLANG_VERSION_MAJOR >= 4)
      std::move(Invocation),
#else
      Invocation,
#endif
      std::make_shared<clang::PCHContainerOperations>(),
      Diagnostics,
      action));
  assert(unit);

  return std::move(unit);
}
