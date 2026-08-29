#include <clang-c-frontend/clang_ast_dump.h>

#include <clang/AST/ASTContext.h>
#include <clang/Basic/Diagnostic.h>
#include <llvm/Support/raw_ostream.h>
#include <util/config/config.h>

void enable_ast_dump_colors(llvm::raw_ostream &os, clang::ASTContext &ctx)
{
  /* command_line_options.cpp resolves --color (auto honours isatty on stderr)
   * before the frontend runs. The dump lands in a std::string that the logger
   * may write anywhere, so an unconditional opt-in would leak escapes into
   * redirected output. */
  const bool colored = config.options.get_bool_option("color");

  os.enable_colors(colored);
  ctx.getDiagnostics().setShowColors(colored);
}
