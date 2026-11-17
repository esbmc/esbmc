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

expr2tc parse_integer(clang::NumericLiteralParser &parser, bool neg)
{
  llvm::APInt val(64, 0);
  if (parser.GetIntegerValue(val))
    return expr2tc();

  const BigInt n(val.getZExtValue());
  const std::pair<type2tc, type2tc> candidates[] = {
    {int_type2(), uint_type2()},
    {long_int_type2(), long_uint_type2()},
    {long_long_int_type2(), long_long_uint_type2()}};
  const size_t first = parser.isLongLong ? 2 : parser.isLong ? 1 : 0;
  // C11 6.4.4.1: decimal literals without U only try signed types.
  for (size_t i = first; i < 3; ++i)
  {
    const auto &[signed_type, unsigned_type] = candidates[i];
    if (!parser.isUnsigned && n < BigInt::power2(signed_type->get_width() - 1))
      return from_integer(neg ? -n : n, signed_type);
    if (
      (parser.isUnsigned || parser.getRadix() != 10) &&
      n < BigInt::power2(unsigned_type->get_width()))
      return from_integer(neg ? -n : n, unsigned_type);
  }
  return expr2tc();
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

  return parse_integer(parser, neg);
}

expr2tc parse_constraint_value(ParseState &ps)
{
  expr2tc value = parse_number(ps);
  if (!value || !ps.is(clang::tok::minus))
    return value;

  // The witness writer spells signed minima as -MAX - 1 (C11 6.4.4.1).
  if (
    !is_constant_int2t(value) || !is_signedbv_type(value->type) ||
    to_constant_int2t(value).value !=
      1 - BigInt::power2(value->type->get_width() - 1))
    return expr2tc();
  ps.advance();
  const expr2tc one = parse_number(ps);
  if (
    !one || !is_constant_int2t(one) || one->type != int_type2() ||
    to_constant_int2t(one).value != 1)
    return expr2tc();
  return from_integer(to_constant_int2t(value).value - 1, value->type);
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

  expr2tc rhs = parse_constraint_value(ps);
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
