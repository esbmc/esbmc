#include <util/base/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/AST/ASTImporter.h>
#include <clang/Basic/Version.inc>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#if CLANG_VERSION_MAJOR >= 22
#  include <clang/Options/Options.h>
#else
#  include <clang/Driver/Options.h>
#endif
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Option/ArgList.h>
#if CLANG_VERSION_MAJOR < 16
#  include <llvm/Support/Host.h>
#else
#  include <llvm/TargetParser/Host.h>
#endif
#include <llvm/Support/Path.h>
CC_DIAGNOSTIC_POP()

#include <clang-c-frontend/AST/build_ast.h>
#include <clang-c-frontend/AST/vfs_adapter.h>
#include <clang-c-frontend/AST/vfs_paths.h>

/// Builds a clang driver initialized for running clang tools.
static clang::driver::Driver *newDriver(
  clang::DiagnosticsEngine *Diagnostics,
  const char *BinaryName,
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
{
  clang::driver::Driver *CompilerDriver;
/* Clang's API changed between version 11 and 12 in that it now requires a name
 * to construct a Driver. */
#if CLANG_VERSION_MAJOR >= 12
  CompilerDriver = new clang::driver::Driver(
    BinaryName,
    llvm::sys::getDefaultTargetTriple(),
    *Diagnostics,
    "clang_based_tool",
    std::move(VFS));
#else
  CompilerDriver = new clang::driver::Driver(
    BinaryName,
    llvm::sys::getDefaultTargetTriple(),
    *Diagnostics,
    std::move(VFS));
  CompilerDriver->setTitle("clang_based_tool");
#endif
  return CompilerDriver;
}

std::unique_ptr<clang::ASTUnit> buildASTs(
  const std::string &intrinsics,
  const std::vector<std::string> &compiler_args)
{
  // Bundled headers are served from .rodata; everything else from disk.
  llvm::IntrusiveRefCntPtr<clang::FileManager> Files(
    new clang::FileManager(clang::FileSystemOptions(), esbmc_clang_vfs()));

  // Create everything needed to create a CompilerInvocation,
  // copied from ToolInvocation::run

#if CLANG_VERSION_MAJOR >= 21
  using DiagOptsType = std::shared_ptr<clang::DiagnosticOptions>;
#else
  using DiagOptsType = llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions>;
#endif

  DiagOptsType DiagOpts;
#if CLANG_VERSION_MAJOR >= 21
  DiagOpts = std::make_shared<clang::DiagnosticOptions>();
#else
  DiagOpts = new clang::DiagnosticOptions();
#endif

  std::vector<const char *> Argv;
  for (const std::string &Str : compiler_args)
    Argv.push_back(Str.c_str());
  const char *const BinaryName = Argv[0];

  unsigned MissingArgIndex, MissingArgCount;
#if CLANG_VERSION_MAJOR >= 22
  llvm::opt::InputArgList ParsedArgs = clang::getDriverOptTable().ParseArgs(
    llvm::ArrayRef<const char *>(Argv).slice(1),
    MissingArgIndex,
    MissingArgCount);

#else
  llvm::opt::InputArgList ParsedArgs =
    clang::driver::getDriverOptTable().ParseArgs(
      llvm::ArrayRef<const char *>(Argv).slice(1),
      MissingArgIndex,
      MissingArgCount);

#endif

  clang::ParseDiagnosticArgs(*DiagOpts, ParsedArgs);

  clang::TextDiagnosticPrinter DiagnosticPrinter(
    llvm::errs(),
#if CLANG_VERSION_MAJOR >= 21
    *DiagOpts
#else
    &*DiagOpts
#endif
  );

  clang::DiagnosticsEngine *Diagnostics = new clang::DiagnosticsEngine(
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()),
#if CLANG_VERSION_MAJOR >= 21
    *DiagOpts,
#else
    &*DiagOpts,
#endif
    &DiagnosticPrinter,
    false);

  const std::unique_ptr<clang::driver::Driver> Driver(
    newDriver(Diagnostics, BinaryName, &Files->getVirtualFileSystem()));

  // Since the input might only be virtual, don't check whether it exists.
  Driver->setCheckInputsExist(false);
  const std::unique_ptr<clang::driver::Compilation> Compilation(
    Driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));

  const clang::driver::JobList &Jobs = Compilation->getJobs();
  assert(Jobs.size() == 1);

  const llvm::opt::ArgStringList *const CC1Args = &Jobs.begin()->getArguments();

  std::shared_ptr<clang::CompilerInvocation> Invocation(
    clang::tooling::newInvocation(Diagnostics, *CC1Args, BinaryName));

  // Show the invocation, with -v.
  if (Invocation->getHeaderSearchOpts().Verbose)
  {
    llvm::errs() << "clang Invocation:\n";
    Compilation->getJobs().Print(llvm::errs(), "\n", true);
    llvm::errs() << "\n";
  }

  /* Inject ESBMC's intrinsics as a forced include ahead of any the user asked
   * for. They must land after the builtin #defines they rely on but before the
   * first forced include, which can transitively reach ESBMC's own models and
   * would otherwise see nondet_* / __ESBMC_* undeclared (github #5868).
   * Forced includes are emitted into the predefines buffer in order, after all
   * -D macros, so being first satisfies both constraints.
   *
   * This is what lets the ASTUnit be built via LoadFromCompilerInvocation,
   * which -- unlike the ...Action() overload -- accepts our FileManager and so
   * reads bundled headers through esbmc_clang_vfs(). */
  const std::string intrinsics_path = clang_vfs_root() + "/esbmc_intrinsics.h";
  clang::PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
  /* getMemBufferCopy, not getMemBuffer: `intrinsics` belongs to the caller and
   * does not outlive the returned ASTUnit. Ownership of the copy passes to
   * clang, which frees it (RetainRemappedFileBuffers defaults to false). */
  PPOpts.addRemappedFile(
    intrinsics_path,
    llvm::MemoryBuffer::getMemBufferCopy(intrinsics, intrinsics_path)
      .release());
  PPOpts.Includes.insert(PPOpts.Includes.begin(), intrinsics_path);

  // Create ASTUnit
  std::unique_ptr<clang::ASTUnit> unit(
    clang::ASTUnit::LoadFromCompilerInvocation(
      std::move(Invocation),
      std::make_shared<clang::PCHContainerOperations>(),
#if CLANG_VERSION_MAJOR >= 21
      DiagOpts,
#endif
      Diagnostics,
      /* Raw pointer: clang 21 and earlier declare this parameter as
       * FileManager *, clang 22 as IntrusiveRefCntPtr<FileManager>, and the
       * latter converts implicitly from the former (retaining, so the returned
       * ASTUnit keeps the manager alive past this scope). */
      Files.get()));
  assert(unit);

  return unit;
}

/// Import one decl from the source context into \p Importer's destination.
/// Returns the imported decl, or nullptr if the import failed (the error is
/// logged and consumed so the rest of the merge can proceed).
///
/// clang::ASTImporter::Import() imports the decl node itself but does NOT
/// recurse into the children of a DeclContext such as a LinkageSpecDecl
/// (`extern "C" { ... }`). Importing the wrapper alone yields an empty
/// linkage-spec, so the functions inside it silently vanish from the merged
/// AST. Every function in ESBMC's C++ operational models is declared inside an
/// `extern "C"` block, which is why merging a second C++ translation unit
/// dropped all of its symbols. C functions are direct children of the
/// translation unit, so they import individually and were unaffected.
///
/// We therefore import each child of a LinkageSpecDecl explicitly and attach it
/// to the imported wrapper's context.
static clang::Decl *
importDecl(clang::ASTImporter &Importer, clang::Decl *FromDecl)
{
  llvm::Expected<clang::Decl *> ImportedOrErr = Importer.Import(FromDecl);
  if (!ImportedOrErr)
  {
    llvm::Error Err = ImportedOrErr.takeError();
    llvm::errs() << "Error: " << Err << "\n";
    consumeError(std::move(Err));
    return nullptr;
  }
  clang::Decl *Imported = *ImportedOrErr;

  auto *FromLS = llvm::dyn_cast<clang::LinkageSpecDecl>(FromDecl);
  auto *ToLS = llvm::dyn_cast_or_null<clang::LinkageSpecDecl>(Imported);
  if (FromLS && ToLS)
    for (clang::Decl *Child : FromLS->decls())
      if (clang::Decl *ImportedChild = importDecl(Importer, Child))
        if (ImportedChild->getLexicalDeclContext() != ToLS)
          ToLS->addDecl(ImportedChild);

  return Imported;
}

void mergeASTs(
  const std::unique_ptr<clang::ASTUnit> &FromUnit,
  std::unique_ptr<clang::ASTUnit> &ToUnit)
{
  // Call enableSourceFileDiagnostics on the
  // ASTUnit objects to get diagnostics.
  FromUnit->enableSourceFileDiagnostics();
  ToUnit->enableSourceFileDiagnostics();

  clang::ASTImporter Importer(
    ToUnit->getASTContext(),
    ToUnit->getFileManager(),
    FromUnit->getASTContext(),
    FromUnit->getFileManager(),
    false);

  Importer.setODRHandling(clang::ASTImporter::ODRHandlingType::Liberal);

  for (auto decl : FromUnit->getASTContext().getTranslationUnitDecl()->decls())
    importDecl(Importer, decl);
}
