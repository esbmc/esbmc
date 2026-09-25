#ifndef CLANG_C_FRONTEND_CLANG_AST_DUMP_H
#define CLANG_C_FRONTEND_CLANG_AST_DUMP_H

namespace clang
{
class ASTContext;
}
namespace llvm
{
class raw_ostream;
}

/// Opt @p os into clang's coloured AST dumps, honouring `--color`.
///
/// Two independent gates suppress the colour that `--parse-tree-*` already
/// gets (esbmc/esbmc#746): `Decl::dump` builds its ASTDumper from
/// `ASTContext::getDiagnostics().getShowColors()`, and `raw_ostream` refuses
/// to emit escapes to a stream that is not a terminal -- which the
/// `raw_os_ostream` wrappers around `std::ostringstream` never are. Open both,
/// or neither.
void enable_ast_dump_colors(llvm::raw_ostream &os, clang::ASTContext &ctx);

#endif
