#include <ld-frontend/ir_gen/st_fb_translator.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <cctype>
#include <stdexcept>
#include <string>

// Configuration toggle for reproducing the two paper configurations from one
// binary.  Selected by the --ld-sound-mode CLI flag so the chosen semantics is
// visible in the verification record (rather than an undiscoverable env var):
//   default (flag unset) -> "analog-extended": over-approximate unsupported ST
//                           constructs (function calls, member access) as
//                           nondeterministic, enabling analog programs (SWaT) to
//                           be modelled at the cost of soundness (RQ6).
//   --ld-sound-mode      -> "sound Boolean/integer": such constructs throw, so
//                           the FB body falls back to a no-op; no over-
//                           approximation, zero false positives (RQ2, RQ5).
static bool sound_mode()
{
  return config.options.get_bool_option("ld-sound-mode");
}

// Scan-watchdog instrumentation is opt-in: it injects an assertion and thus
// changes the verified model (see the WHILE handler).  --ld-scan-budget bounds
// the tolerated loop iterations (default 8).
static bool watchdog_enabled()
{
  return config.options.get_bool_option("ld-scan-watchdog");
}
static long long scan_budget()
{
  const std::string s = config.options.get_option("ld-scan-budget");
  if (!s.empty())
  {
    try
    {
      const long long v = std::stoll(s);
      if (v > 0)
        return v;
    }
    catch (const std::exception &)
    {
      // fall through to the default on a malformed value
    }
  }
  return 8;
}
static void require_tolerant(const char *what)
{
  if (sound_mode())
    throw std::runtime_error(
      std::string("st_fb_translator: ") + what +
      " requires analog-extended mode (do not pass --ld-sound-mode)");
}

// -----------------------------------------------------------------------
// Lexer
// -----------------------------------------------------------------------

void st_fb_translator::skip_ws()
{
  while (pos_ < src_.size())
  {
    char c = src_[pos_];
    if (std::isspace(static_cast<unsigned char>(c)))
    {
      ++pos_;
      continue;
    }
    // ST block comment (* ... *)
    if (c == '(' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '*')
    {
      pos_ += 2;
      while (pos_ + 1 < src_.size() &&
             !(src_[pos_] == '*' && src_[pos_ + 1] == ')'))
        ++pos_;
      pos_ += 2;
      continue;
    }
    break;
  }
}

bool st_fb_translator::eof()
{
  skip_ws();
  return pos_ >= src_.size();
}

static std::string lower(std::string s)
{
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string st_fb_translator::peek_word()
{
  size_t save = pos_;
  std::string w = next_word();
  pos_ = save;
  return lower(w);
}

std::string st_fb_translator::next_word()
{
  skip_ws();
  size_t start = pos_;
  if (pos_ < src_.size() &&
      (std::isalpha(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_'))
  {
    while (pos_ < src_.size() &&
           (std::isalnum(static_cast<unsigned char>(src_[pos_])) ||
            src_[pos_] == '_'))
      ++pos_;
  }
  return src_.substr(start, pos_ - start);
}

bool st_fb_translator::accept_kw(const char *kw)
{
  size_t save = pos_;
  std::string w = lower(next_word());
  if (w == kw)
    return true;
  pos_ = save;
  return false;
}

void st_fb_translator::expect_kw(const char *kw)
{
  if (!accept_kw(kw))
    throw std::runtime_error(std::string("st_fb_translator: expected '") + kw +
                             "'");
}

bool st_fb_translator::accept_sym(const char *s)
{
  skip_ws();
  size_t len = std::string(s).size();
  if (src_.compare(pos_, len, s) == 0)
  {
    pos_ += len;
    return true;
  }
  return false;
}

void st_fb_translator::expect_sym(const char *s)
{
  if (!accept_sym(s))
    throw std::runtime_error(std::string("st_fb_translator: expected '") + s +
                             "'");
}

// -----------------------------------------------------------------------
// Parser -> exprt
// -----------------------------------------------------------------------

exprt st_fb_translator::parse_primary()
{
  skip_ws();
  // integer or REAL literal
  if (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_])))
  {
    size_t start = pos_;
    while (pos_ < src_.size() &&
           std::isdigit(static_cast<unsigned char>(src_[pos_])))
      ++pos_;
    bool is_real = false;
    if (pos_ + 1 < src_.size() && src_[pos_] == '.' &&
        std::isdigit(static_cast<unsigned char>(src_[pos_ + 1])))
    {
      is_real = true;
      ++pos_;
      while (pos_ < src_.size() &&
             std::isdigit(static_cast<unsigned char>(src_[pos_])))
        ++pos_;
    }
    std::string num = src_.substr(start, pos_ - start);
    if (is_real)
      return from_double(std::stod(num), double_type());
    return from_integer(BigInt(std::stoll(num)), int_type());
  }
  // parenthesised
  if (accept_sym("("))
  {
    exprt e = parse_expr();
    expect_sym(")");
    return e;
  }
  // identifier / boolean literal / function call / member access
  std::string w = next_word();
  if (w.empty())
    throw std::runtime_error("st_fb_translator: expected expression");
  std::string lw = lower(w);
  if (lw == "true")
    return true_exprt();
  if (lw == "false")
    return false_exprt();
  // function call  IDENT(args...) : over-approximate result as nondeterministic
  if (accept_sym("("))
  {
    require_tolerant("function-call result");
    int depth = 1;
    while (pos_ < src_.size() && depth > 0)
    {
      if (src_[pos_] == '(') ++depth;
      else if (src_[pos_] == ')') --depth;
      ++pos_;
    }
    return side_effect_expr_nondett(int_type());
  }
  // member access  IDENT.FIELD (e.g. timer.Q) : nondeterministic Boolean
  if (accept_sym("."))
  {
    require_tolerant("member access");
    next_word(); // consume field name
    return side_effect_expr_nondett(typet("bool"));
  }
  return resolve_(w); // typed symbol_exprt
}

// Build a binary arithmetic node, deriving its result type from the operands:
// promote to double when either side is floating point (REAL/analog), otherwise
// stay integer.  Hard-coding int here would silently truncate the analog (SWaT)
// arithmetic this frontend now supports.
static bool is_floating(const typet &t)
{
  return t.id() == "floatbv" || t.id() == "fixedbv";
}
// Floating-point arithmetic must use the ieee_* operators: the LD frontend has
// no adjust pass to rewrite plus/mult on a floatbv operand, and the simplifier
// asserts on a plain plus/mult node whose type is floatbv.  The irep2 migration
// defaults the rounding mode.
static irep_idt arith_id(const irep_idt &op, bool real)
{
  if (!real)
    return op;
  if (op == exprt::plus)
    return "ieee_add";
  if (op == exprt::minus)
    return "ieee_sub";
  if (op == exprt::mult)
    return "ieee_mul";
  if (op == exprt::div)
    return "ieee_div";
  return op;
}
static exprt make_binary_arith(const irep_idt &op, exprt lhs, exprt rhs)
{
  const bool real = is_floating(lhs.type()) || is_floating(rhs.type());
  const typet result = real ? double_type() : int_type();
  if (lhs.type() != result)
    lhs = typecast_exprt(lhs, result);
  if (rhs.type() != result)
    rhs = typecast_exprt(rhs, result);
  exprt e(arith_id(op, real), result);
  e.copy_to_operands(lhs, rhs);
  return e;
}

// arithmetic with IEC 61131-3 precedence: '*' and '/' bind tighter than '+' and
// '-'; both levels are left-associative.  Modelling precedence keeps a mixed-
// operator guard such as `a + b * c` from mis-parsing as `(a + b) * c`.
exprt st_fb_translator::parse_expr()
{
  exprt lhs = parse_term();
  for (;;)
  {
    irep_idt op;
    if (accept_sym("+"))
      op = exprt::plus;
    else if (accept_sym("-"))
      op = exprt::minus;
    else
      break;
    lhs = make_binary_arith(op, lhs, parse_term());
  }
  return lhs;
}

exprt st_fb_translator::parse_term()
{
  exprt lhs = parse_primary();
  for (;;)
  {
    irep_idt op;
    if (accept_sym("*"))
      op = exprt::mult;
    else if (accept_sym("/"))
      op = exprt::div;
    else
      break;
    lhs = make_binary_arith(op, lhs, parse_primary());
  }
  return lhs;
}

// Promote a comparison's operands to a common type: if either side is REAL
// (floating), cast the other to double so a mixed REAL/INT comparison does not
// build a relation node with mismatched operand sorts (the LD frontend has no
// adjust pass to fix it up later).
static void promote_numeric(exprt &a, exprt &b)
{
  if (is_floating(a.type()) && !is_floating(b.type()))
    b = typecast_exprt(b, double_type());
  else if (is_floating(b.type()) && !is_floating(a.type()))
    a = typecast_exprt(a, double_type());
}

// value possibly followed by a relational operator (=, <>, <, <=, >, >=)
exprt st_fb_translator::parse_condition()
{
  exprt lhs = parse_expr();
  irep_idt relop;
  bool equality = false, negate = false;
  // multi-char operators first
  if (accept_sym("<="))
    relop = "<=";
  else if (accept_sym(">="))
    relop = ">=";
  else if (accept_sym("<>"))
    equality = negate = true;
  else if (accept_sym("<"))
    relop = "<";
  else if (accept_sym(">"))
    relop = ">";
  else if (accept_sym("="))
    equality = true;
  else
    return lhs;

  exprt rhs = parse_expr();
  promote_numeric(lhs, rhs);
  if (equality)
  {
    exprt eq = equality_exprt(lhs, rhs);
    return negate ? static_cast<exprt>(not_exprt(eq)) : eq;
  }
  return binary_relation_exprt(lhs, relop, rhs);
}

// -----------------------------------------------------------------------
// Parser -> codet
// -----------------------------------------------------------------------

codet st_fb_translator::parse_stmt()
{
  std::string kw = peek_word();

  if (kw == "if")
  {
    expect_kw("if");
    exprt cond = parse_condition();
    expect_kw("then");
    code_blockt then_blk;
    static const char *const term_if[] = {"end_if", "else", "elsif", nullptr};
    parse_stmt_list(then_blk, term_if);

    code_ifthenelset ite;
    ite.cond() = cond;
    ite.then_case() = then_blk;

    if (accept_kw("else"))
    {
      code_blockt else_blk;
      static const char *const term_else[] = {"end_if", nullptr};
      parse_stmt_list(else_blk, term_else);
      ite.else_case() = else_blk;
    }
    expect_kw("end_if");
    accept_sym(";");
    return ite;
  }

  if (kw == "while")
  {
    expect_kw("while");
    exprt cond = parse_condition();
    expect_kw("do");
    code_blockt body;
    static const char *const term_w[] = {"end_while", nullptr};
    parse_stmt_list(body, term_w);
    expect_kw("end_while");
    accept_sym(";");

    code_whilet loop;
    loop.cond() = cond;

    // Without the scan-watchdog, the WHILE stays a plain loop: a non-terminating
    // Ladder Logic Bomb is then caught by ESBMC's unwinding assertion (--unwind),
    // and no extra assertion is added to the verified model.
    if (!watchdog_enabled())
    {
      loop.body() = body;
      return loop;
    }

    // Scan-watchdog instrumentation (opt-in via --ld-scan-watchdog).  A real PLC
    // trips a watchdog timer if a scan overruns; a non-terminating rung loop is
    // exactly such an overrun.  We prepend a per-loop iteration counter and
    // assert it stays within the scan budget, turning a (trigger-gated)
    // non-terminating Ladder Logic Bomb into a reachable safety violation that
    // incremental BMC detects, while bounded legitimate loops within budget
    // remain SAFE.  This injects an assertion, so it deliberately changes the
    // verified model and is therefore gated behind the explicit flag.  The
    // budget (--ld-scan-budget, default 8) bounds tolerated iterations and
    // should be kept <= the BMC --unwind so the assertion is reachable.
    const long long budget = scan_budget();
    symbol_exprt wd = resolve_("__wd" + std::to_string(wd_counter_++));

    code_blockt instrumented;
    exprt inc(exprt::plus, wd.type());
    inc.copy_to_operands(wd, from_integer(BigInt(1), wd.type()));
    instrumented.copy_to_operands(code_assignt(wd, inc));
    instrumented.copy_to_operands(code_assertt(binary_relation_exprt(
      wd, "<=", from_integer(BigInt(budget), wd.type()))));
    for (const auto &op : body.operands())
      instrumented.copy_to_operands(static_cast<const codet &>(op));

    loop.body() = instrumented;

    code_blockt out;
    out.copy_to_operands(code_assignt(wd, from_integer(BigInt(0), wd.type())));
    out.copy_to_operands(loop);
    return out;
  }

  // VAR / VAR_INPUT / ... END_VAR declaration block embedded in the body -> skip
  if (kw == "var" || kw.rfind("var_", 0) == 0)
  {
    require_tolerant("embedded VAR block");
    next_word();
    while (!eof())
    {
      if (peek_word() == "end_var") { next_word(); break; }
      if (next_word().empty()) ++pos_; // advance over a symbol/token
    }
    accept_sym(";");
    return code_blockt();
  }

  // assignment: IDENT := value ;
  std::string name = next_word();
  if (name.empty())
    throw std::runtime_error("st_fb_translator: expected statement");
  // call statement  IDENT(args...) ;  (e.g. a timer/FB invocation) -> no-op
  if (accept_sym("("))
  {
    require_tolerant("call statement");
    int depth = 1;
    while (pos_ < src_.size() && depth > 0)
    {
      if (src_[pos_] == '(') ++depth;
      else if (src_[pos_] == ')') --depth;
      ++pos_;
    }
    accept_sym(";");
    return code_skipt();
  }
  // member-access assignment target  IDENT.FIELD := ... ; -> over-approximate
  if (accept_sym("."))
  {
    require_tolerant("member-access assignment");
    next_word();
    if (accept_sym(":="))
    {
      parse_condition();
    }
    accept_sym(";");
    return code_skipt();
  }
  // not an assignment (e.g. a declaration  IDENT : TYPE [:= v];) -> skip stmt
  if (!accept_sym(":="))
  {
    require_tolerant("non-assignment statement");
    while (!eof() && !accept_sym(";"))
      if (next_word().empty()) ++pos_;
    return code_blockt();
  }
  symbol_exprt lhs = resolve_(name);
  exprt rhs = parse_condition();
  accept_sym(";");
  if (rhs.type() != lhs.type())
    rhs = typecast_exprt(rhs, lhs.type());
  return code_assignt(lhs, rhs);
}

void st_fb_translator::parse_stmt_list(
  code_blockt &out,
  const char *const *terminators)
{
  for (;;)
  {
    if (eof())
      break;
    std::string w = peek_word();
    bool is_term = false;
    for (const char *const *t = terminators; *t; ++t)
      if (w == *t)
      {
        is_term = true;
        break;
      }
    if (is_term)
      break;
    out.copy_to_operands(parse_stmt());
  }
}

code_blockt st_fb_translator::translate(const std::string &body)
{
  src_ = body;
  pos_ = 0;
  code_blockt out;
  static const char *const term_none[] = {nullptr};
  parse_stmt_list(out, term_none);
  return out;
}
