#include <util/base/compiler_defs.h>
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
#include <util/arith/arith_tools.h>
#include <util/arith/ieee_float.h>
#include <util/lang/c_types.h>
#include <util/message/message.h>
#include <string_view>

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
    diag_ids,
    diag_opts,
    new clang::IgnoringDiagConsumer(),
    true};
  clang::FileSystemOptions fs_opts;
  clang::FileManager fm{fs_opts};
  clang::SourceManager sm{diags, fm};
  std::unique_ptr<clang::TargetInfo> target_info;

  LexerContext()
  {
    lo.C17 = 1;
    lo.GNUMode = 1;
    // CreateTargetInfo takes TargetOptions by reference since Clang 21; older
    // releases (e.g. Clang 18) expect a const std::shared_ptr<TargetOptions>&.
#if CLANG_VERSION_MAJOR >= 21
    clang::TargetOptions target_opts;
    target_opts.Triple = llvm::sys::getDefaultTargetTriple();
    target_info.reset(clang::TargetInfo::CreateTargetInfo(diags, target_opts));
#else
    auto target_opts = std::make_shared<clang::TargetOptions>();
    target_opts->Triple = llvm::sys::getDefaultTargetTriple();
    target_info.reset(clang::TargetInfo::CreateTargetInfo(diags, target_opts));
#endif
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
    : ParseState(ctx_, src, ctx_.lo)
  {
  }

  ParseState(
    clang_c_lexert::LexerContext &ctx_,
    const std::string &src,
    const clang::LangOptions &options,
    bool retain_comments = false)
    : ctx(ctx_),
      lex(
        clang::SourceLocation(),
        options,
        src.data(),
        src.data(),
        src.data() + src.size())
  {
    lex.SetCommentRetentionState(retain_comments);
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

/// A decimal floating literal such as 2.5595000e+2f, rounded to the type its
/// suffix names.
expr2tc parse_floating(
  llvm::StringRef spelling,
  const clang::NumericLiteralParser &parser,
  bool neg)
{
  const type2tc type = parser.isFloat  ? float_type2()
                       : parser.isLong ? long_double_type2()
                                       : double_type2();
  const std::string lower = spelling.lower();
  const llvm::StringRef digits = llvm::StringRef(lower).rtrim("fl");
  // Other suffixes (q, f16, i) and hex floats are not read.
  if (
    parser.getRadix() != 10 || !is_floatbv_type(type) ||
    digits.find_first_not_of("0123456789.e+-") != llvm::StringRef::npos)
    return expr2tc();

  auto [mantissa, exp_str] = digits.split('e');
  exp_str.consume_front("+");
  // Beyond any format's range; from_base10 would compute 5^exponent.
  if (exp_str.ltrim('-').size() > 4)
    return expr2tc();
  const auto [whole, frac] = mantissa.split('.');

  BigInt exponent = exp_str.empty() ? BigInt(0) : string2integer(exp_str.str());
  exponent -= frac.size();
  BigInt significand = string2integer(whole.str() + frac.str());
  if (neg)
    significand = -significand;

  ieee_floatt value{ieee_float_spect(to_floatbv_type(type))};
  value.from_base10(significand, exponent);
  return constant_floatbv2tc(value);
}

/// The INFINITY macro: a float constant (C11 7.12p4).
expr2tc parse_infinity(ParseState &ps, bool neg)
{
  if (
    !ps.is(clang::tok::raw_identifier) ||
    ps.cur.getRawIdentifier() != "INFINITY" || !is_floatbv_type(float_type2()))
    return expr2tc();
  ps.advance();
  ieee_floatt inf = ieee_floatt::plus_infinity(
    ieee_float_spect(to_floatbv_type(float_type2())));
  if (neg)
    inf.negate();
  return constant_floatbv2tc(inf);
}

expr2tc parse_number(ParseState &ps)
{
  bool neg = false;
  if (ps.is(clang::tok::minus) || ps.is(clang::tok::plus))
  {
    neg = ps.is(clang::tok::minus);
    ps.advance();
  }
  if (!ps.is(clang::tok::numeric_constant))
    return parse_infinity(ps, neg);

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

  if (parser.hadError)
    return expr2tc();
  if (parser.isFloatingLiteral())
    return parse_floating(spelling, parser, neg);
  if (!parser.isIntegerLiteral())
    return expr2tc();

  llvm::APInt val(64, 0);
  parser.GetIntegerValue(val);

  BigInt n =
    parser.isUnsigned ? BigInt(val.getZExtValue()) : BigInt(val.getSExtValue());
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

  // '\result' lexes as tok::unknown ('\') followed by tok::raw_identifier
  // ('result').
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

  expr2tc rhs = parse_number(ps);
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

namespace
{
class control_body_scannert
{
  const std::string &source;
  const std::set<size_t> &invariant_lines;
  ParseState ps;
  std::map<size_t, size_t> body_ends;
  size_t offset = 0, line = 1;
  size_t last_end = 0, depth = 0;
  size_t line_start = 0, comment_end = 0;
  bool at_line_start = false;
  bool valid;

  bool locate_token()
  {
    at_line_start = ps.cur.isAtStartOfLine();
    while (ps.is(clang::tok::comment))
    {
      const size_t start = ps.cur.getLocation().getRawEncoding();
      const std::string_view comment(source.data() + start, ps.cur.getLength());
      if (comment.find('\n') != std::string_view::npos)
        comment_end = start + comment.size();
      ps.advance();
      at_line_start |= ps.cur.isAtStartOfLine();
    }
    while (offset < ps.cur.getLocation().getRawEncoding())
      if (source[offset++] == '\n')
      {
        ++line;
        line_start = offset;
      }
    at_line_start &= line_start >= comment_end;
    // Raw lexing cannot account for macros or conditional preprocessing.
    return !ps.is(clang::tok::hash) && !ps.cur.needsCleaning();
  }

  bool advance()
  {
    last_end = ps.cur.getLocation().getRawEncoding() + ps.cur.getLength();
    ps.advance();
    return valid = locate_token();
  }

  bool consume(clang::tok::TokenKind kind)
  {
    return ps.is(kind) && advance();
  }

  bool word(const char *text) const
  {
    return source.compare(
             ps.cur.getLocation().getRawEncoding(), ps.cur.getLength(), text) ==
           0;
  }

  bool control() const
  {
    return word("if") || word("for") || word("while") || word("do");
  }

  bool item()
  {
    if (control())
      return statement();
    if (
      ps.cur.isOneOf(
        clang::tok::l_paren, clang::tok::l_square, clang::tok::l_brace))
      return group();
    if (
      ps.cur.isOneOf(
        clang::tok::eof,
        clang::tok::r_paren,
        clang::tok::r_square,
        clang::tok::r_brace))
      return false;
    return advance();
  }

  bool sequence(clang::tok::TokenKind close)
  {
    while (valid && !ps.is(close))
      if (!item())
        return false;
    return valid && (close == clang::tok::eof || advance());
  }

  bool group()
  {
    if (depth == 1024)
      return false;
    const auto close = ps.is(clang::tok::l_paren)    ? clang::tok::r_paren
                       : ps.is(clang::tok::l_square) ? clang::tok::r_square
                                                     : clang::tok::r_brace;
    ++depth;
    const bool result = advance() && sequence(close);
    --depth;
    return result;
  }

  bool body()
  {
    const size_t start_line = line;
    const bool wrap =
      !ps.is(clang::tok::l_brace) && invariant_lines.count(line);
    if (wrap && !at_line_start)
      return false;
    if (!statement())
      return false;
    if (wrap)
      body_ends[start_line] = last_end;
    return true;
  }

  bool conditional()
  {
    const bool is_if = word("if");
    if (!advance() || !ps.is(clang::tok::l_paren) || !group() || !body())
      return false;
    return !is_if || !word("else") || (advance() && body());
  }

  bool do_statement()
  {
    return advance() && body() && word("while") && advance() &&
           ps.is(clang::tok::l_paren) && group() && consume(clang::tok::semi);
  }

  bool simple_statement()
  {
    if (word("case") || word("default"))
      return false;
    bool first = true;
    while (valid && !ps.is(clang::tok::semi))
    {
      if (control() || word("switch") || !item())
        return false;
      if (first && ps.is(clang::tok::colon))
        return false;
      first = false;
    }
    return consume(clang::tok::semi);
  }

  bool statement()
  {
    if (!valid || depth == 1024)
      return false;
    ++depth;
    bool result;
    if (ps.is(clang::tok::l_brace))
      result = group();
    else if (word("do"))
      result = do_statement();
    else if (control())
      result = conditional();
    else
      result = simple_statement();
    --depth;
    return result;
  }

public:
  control_body_scannert(
    clang_c_lexert::LexerContext &ctx,
    const std::string &source,
    const std::set<size_t> &invariant_lines,
    const clang::LangOptions &options)
    : source(source),
      invariant_lines(invariant_lines),
      ps(ctx, source, options, true),
      valid(locate_token())
  {
  }

  std::optional<std::map<size_t, size_t>> scan()
  {
    if (!sequence(clang::tok::eof))
      return std::nullopt;
    return body_ends;
  }
};
} // namespace

std::optional<std::map<size_t, size_t>> clang_c_lexert::find_control_body_ends(
  const std::string &source,
  const std::set<size_t> &invariant_lines)
{
  auto options = ctx->lo;
  options.Digraphs = true;
  options.Trigraphs = true;
  return control_body_scannert(*ctx, source, invariant_lines, options).scan();
}
