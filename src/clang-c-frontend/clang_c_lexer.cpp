#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Basic/Version.inc>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/TokenKinds.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/LiteralSupport.h>
#include <clang/Lex/Token.h>
#include <llvm/ADT/APInt.h>
#if CLANG_VERSION_MAJOR < 16
#  include <llvm/Support/Host.h>
#else
#  include <llvm/TargetParser/Host.h>
#endif
CC_DIAGNOSTIC_POP()

#include <clang-c-frontend/clang_c_lexer.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/message.h>

struct clang_c_lexert::LexerContext
{
  clang::LangOptions lo;
#if CLANG_VERSION_MAJOR >= 21
  clang::DiagnosticOptions diag_opts;
#else
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diag_opts{
    new clang::DiagnosticOptions()};
#endif
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids{
    new clang::DiagnosticIDs()};
  clang::DiagnosticsEngine diags{
    diag_ids, diag_opts, new clang::IgnoringDiagConsumer(), true};
  clang::FileSystemOptions fs_opts;
  clang::FileManager fm{fs_opts};
  clang::SourceManager sm{diags, fm};
  std::unique_ptr<clang::TargetInfo> target_info;

  LexerContext()
  {
    lo.C17 = 1;
    lo.GNUMode = 1;
    clang::TargetOptions target_opts;
    target_opts.Triple = llvm::sys::getDefaultTargetTriple();
    target_info.reset(clang::TargetInfo::CreateTargetInfo(diags, target_opts));
  }
};

namespace
{

struct ParseState
{
  clang_c_lexert::LexerContext &ctx;
  clang::Lexer lex;
  clang::Token cur;

  ParseState(clang_c_lexert::LexerContext &ctx_, const std::string &src)
    : ctx(ctx_),
      lex(
        clang::SourceLocation(),
        ctx_.lo,
        src.data(),
        src.data(),
        src.data() + src.size())
  {
    advance();
  }

  void advance()
  {
    lex.LexFromRawLexer(cur);
  }

  bool is(clang::tok::TokenKind k) const
  {
    return cur.is(k);
  }
};

expr2tc parse_integer(ParseState &ps)
{
  bool neg = false;
  if (ps.is(clang::tok::minus))
  {
    neg = true;
    ps.advance();
  }
  if (!ps.is(clang::tok::numeric_constant))
    return expr2tc();

  clang::Token t = ps.cur;
  ps.advance();

  llvm::StringRef spelling(t.getLiteralData(), t.getLength());
  clang::NumericLiteralParser parser(
    spelling,
    clang::SourceLocation(),
    ps.ctx.sm,
    ps.ctx.lo,
    *ps.ctx.target_info,
    ps.ctx.diags);

  if (parser.hadError || !parser.isIntegerLiteral())
    return expr2tc();

  llvm::APInt val(64, 0);
  parser.GetIntegerValue(val);

  BigInt n = parser.isUnsigned ? BigInt(val.getZExtValue())
                               : BigInt(val.getSExtValue());
  if (neg)
    n = -n;

  return constant_int2tc(get_int64_type(), n);
}

expr2tc do_parse_expr(ParseState &ps);

expr2tc do_parse_comparison(ParseState &ps)
{
  if (ps.is(clang::tok::l_paren))
  {
    ps.advance();
    expr2tc inner = do_parse_expr(ps);
    if (!inner || !ps.is(clang::tok::r_paren))
      return expr2tc();
    ps.advance();
    return inner;
  }

  // '\result' lexes as tok::unknown ('\') followed by tok::raw_identifier ('result').
  if (!ps.is(clang::tok::unknown))
    return expr2tc();
  ps.advance();
  if (!ps.is(clang::tok::raw_identifier))
    return expr2tc();
  if (ps.cur.getRawIdentifier() != "result")
    return expr2tc();
  ps.advance();

  const expr2tc lhs = symbol2tc(get_int64_type(), "\\result");

  // Validate the operator before consuming it or parsing the RHS.
  const clang::tok::TokenKind op = ps.cur.getKind();
  switch (op)
  {
  case clang::tok::equalequal:
  case clang::tok::exclaimequal:
  case clang::tok::less:
  case clang::tok::lessequal:
  case clang::tok::greater:
  case clang::tok::greaterequal:
    break;
  default:
    return expr2tc();
  }
  ps.advance();

  expr2tc rhs = parse_integer(ps);
  if (!rhs)
    return expr2tc();

  switch (op)
  {
  case clang::tok::equalequal:
    return equality2tc(lhs, rhs);
  case clang::tok::exclaimequal:
    return notequal2tc(lhs, rhs);
  case clang::tok::less:
    return lessthan2tc(lhs, rhs);
  case clang::tok::lessequal:
    return lessthanequal2tc(lhs, rhs);
  case clang::tok::greater:
    return greaterthan2tc(lhs, rhs);
  case clang::tok::greaterequal:
    return greaterthanequal2tc(lhs, rhs);
  default:
    log_warning("clang_c_lexer: unhandled operator kind {}", (unsigned)op);
    return expr2tc();
  }
}

expr2tc do_parse_conjunction(ParseState &ps)
{
  expr2tc lhs = do_parse_comparison(ps);
  if (!lhs)
    return lhs;
  while (ps.is(clang::tok::ampamp))
  {
    ps.advance();
    expr2tc rhs = do_parse_comparison(ps);
    if (!rhs)
      return expr2tc();
    lhs = and2tc(lhs, rhs);
  }
  return lhs;
}

expr2tc do_parse_expr(ParseState &ps)
{
  expr2tc lhs = do_parse_conjunction(ps);
  if (!lhs)
    return lhs;
  while (ps.is(clang::tok::pipepipe))
  {
    ps.advance();
    expr2tc rhs = do_parse_conjunction(ps);
    if (!rhs)
      return expr2tc();
    lhs = or2tc(lhs, rhs);
  }
  return lhs;
}

} // namespace

clang_c_lexert::clang_c_lexert() : ctx(std::make_unique<LexerContext>())
{
}

clang_c_lexert::~clang_c_lexert() = default;

expr2tc clang_c_lexert::parse_expr(const std::string &expr)
{
  ParseState ps(*ctx, expr);
  expr2tc e = do_parse_expr(ps);
  if (!e || !ps.is(clang::tok::eof))
    return expr2tc();
  return e;
}
