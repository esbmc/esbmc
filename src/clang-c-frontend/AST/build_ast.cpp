#include <util/base/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/AST/ASTImporter.h>
#include <clang/AST/ASTImporterSharedState.h>
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

  /* Must precede any user -include: those can transitively reach ESBMC's own
   * models, which would then see nondet_* / __ESBMC_* undeclared (#5868). */
  const std::string intrinsics_path = clang_vfs_root() + "/esbmc_intrinsics.h";
  clang::PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
  /* Copy: `intrinsics` does not outlive the ASTUnit. clang frees the buffer,
   * RetainRemappedFileBuffers being false. */
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
      /* Raw pointer: FileManager * up to clang 21, IntrusiveRefCntPtr in 22,
       * which converts implicitly and retains. */
      Files.get()));

  return unit;
}

static void log_import_error(llvm::Error Err)
{
  llvm::errs() << "Error: " << Err << "\n";
  consumeError(std::move(Err));
}

/// The importer maps a variable definition onto one already in the
/// destination and drops its initializer, so the definition merged first
/// wins. A strong definition must override a weak one wherever it appears, as
/// at link time (#7451).
static void override_weak_var(
  clang::ASTImporter &Importer,
  clang::VarDecl &FromVar,
  clang::VarDecl &ToVar)
{
  if (
    FromVar.isThisDeclarationADefinition() == clang::VarDecl::DeclarationOnly ||
    FromVar.hasAttr<clang::WeakAttr>() || !ToVar.hasAttr<clang::WeakAttr>())
    return;

  clang::Expr *Init = nullptr;
  if (FromVar.getInit())
  {
    llvm::Expected<clang::Expr *> InitOrErr =
      Importer.Import(FromVar.getInit());
    if (!InitOrErr)
    {
      log_import_error(InitOrErr.takeError());
      return;
    }
    Init = *InitOrErr;
  }
  ToVar.setInit(Init);
  ToVar.dropAttr<clang::WeakAttr>();
}

/// As override_weak_var, for a function: its parameters map onto the merged
/// decl's, so that the imported body refers to those.
static void override_weak_function(
  clang::ASTImporter &Importer,
  clang::Decl *FromDecl,
  clang::Decl *Imported)
{
  auto *FromFn = llvm::dyn_cast<clang::FunctionDecl>(FromDecl);
  auto *ToFn = llvm::dyn_cast_or_null<clang::FunctionDecl>(Imported);
  if (
    !FromFn || !ToFn || !FromFn->doesThisDeclarationHaveABody() ||
    FromFn->hasAttr<clang::WeakAttr>())
    return;
  const clang::FunctionDecl *ToDef = nullptr;
  if (
    !ToFn->hasBody(ToDef) || !ToDef->hasAttr<clang::WeakAttr>() ||
    ToDef->getNumParams() != FromFn->getNumParams())
    return;

  auto *Def = const_cast<clang::FunctionDecl *>(ToDef);
  for (unsigned i = 0; i < FromFn->getNumParams(); ++i)
    Importer.MapImported(FromFn->getParamDecl(i), Def->getParamDecl(i));
  llvm::Expected<clang::Stmt *> BodyOrErr = Importer.Import(FromFn->getBody());
  if (!BodyOrErr)
  {
    log_import_error(BodyOrErr.takeError());
    return;
  }
  Def->setBody(*BodyOrErr);
  Def->dropAttr<clang::WeakAttr>();
}

/// Applies the overrides to \p FromDecl and, since the importer brings in a
/// namespace's members itself, to every member of a namespace.
static void override_weak(
  clang::ASTImporter &Importer,
  clang::Decl *FromDecl,
  clang::Decl *Imported)
{
  auto *FromVar = llvm::dyn_cast<clang::VarDecl>(FromDecl);
  auto *ToVar = llvm::dyn_cast_or_null<clang::VarDecl>(Imported);
  if (FromVar && ToVar && FromVar->isFileVarDecl())
    override_weak_var(Importer, *FromVar, *ToVar);
  override_weak_function(Importer, FromDecl, Imported);
  if (auto *FromNS = llvm::dyn_cast<clang::NamespaceDecl>(FromDecl))
    for (clang::Decl *Child : FromNS->decls())
      override_weak(Importer, Child, Importer.GetAlreadyImportedOrNull(Child));
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
    log_import_error(ImportedOrErr.takeError());
    return nullptr;
  }
  clang::Decl *Imported = *ImportedOrErr;

  override_weak(Importer, FromDecl, Imported);

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

  /* Without a shared state the importer resolves every name through
   * DeclContext::localUncachedLookup -- a linear, deliberately uncached scan of
   * a destination context that grows with each imported decl, so merging is
   * quadratic and two C++ TUs do not finish (#7556). Supplying the state gives
   * ASTImporter::findDeclsInToCtx its hash-based lookup table instead; the
   * header notes the fallback explicitly ("If not set then the original C/C++
   * lookup is used"). */
  auto SharedState = std::make_shared<clang::ASTImporterSharedState>(
    *ToUnit->getASTContext().getTranslationUnitDecl());

  clang::ASTImporter Importer(
    ToUnit->getASTContext(),
    ToUnit->getFileManager(),
    FromUnit->getASTContext(),
    FromUnit->getFileManager(),
    false,
    SharedState);

  Importer.setODRHandling(clang::ASTImporter::ODRHandlingType::Liberal);

  for (auto decl : FromUnit->getASTContext().getTranslationUnitDecl()->decls())
    importDecl(Importer, decl);
}
