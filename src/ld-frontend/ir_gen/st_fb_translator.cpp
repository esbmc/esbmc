#include <ld-frontend/ir_gen/st_fb_translator.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

// Configuration toggle for reproducing the two paper configurations from one binary:
//   default               -> "analog-extended": over-approximate unsupported ST
//                            constructs (function calls, member access) as
//                            nondeterministic, enabling analog programs (SWaT) to be
//                            modelled at the cost of soundness (RQ6).
//   env LLB_SOUND_MODE set -> "sound Boolean/integer": such constructs throw, so the
//                            FB body falls back to a no-op; no over-approximation,
//                            zero false positives (RQ2, RQ5).
static bool sound_mode()
{
  static const bool s = (std::getenv("LLB_SOUND_MODE") != nullptr);
  return s;
}
static void require_tolerant(const char *what)
{
  if (sound_mode())
    throw std::runtime_error(
      std::string("st_fb_translator: ") + what +
      " requires analog-extended mode (LLB_SOUND_MODE unset)");
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

// arithmetic: left-associative + - * / (precedence not modelled; FB bodies are
// flat single-operator expressions, which is sufficient for the LLB corpus).
exprt st_fb_translator::parse_expr()
{
  exprt lhs = parse_primary();
  for (;;)
  {
    irep_idt op;
    if (accept_sym("+"))
      op = exprt::plus;
    else if (accept_sym("-"))
      op = exprt::minus;
    else if (accept_sym("*"))
      op = exprt::mult;
    else if (accept_sym("/"))
      op = exprt::div;
    else
      break;
    exprt rhs = parse_primary();
    exprt e(op, int_type());
    e.copy_to_operands(lhs, rhs);
    lhs = e;
  }
  return lhs;
}

// value possibly followed by a relational operator (=, <>, <, <=, >, >=)
exprt st_fb_translator::parse_condition()
{
  exprt lhs = parse_expr();
  // multi-char operators first
  if (accept_sym("<="))
    return binary_relation_exprt(lhs, "<=", parse_expr());
  if (accept_sym(">="))
    return binary_relation_exprt(lhs, ">=", parse_expr());
  if (accept_sym("<>"))
    return not_exprt(equality_exprt(lhs, parse_expr()));
  if (accept_sym("<"))
    return binary_relation_exprt(lhs, "<", parse_expr());
  if (accept_sym(">"))
    return binary_relation_exprt(lhs, ">", parse_expr());
  if (accept_sym("="))
    return equality_exprt(lhs, parse_expr());
  return lhs;
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

    // Scan-watchdog instrumentation.  A real PLC trips a watchdog timer if a
    // scan overruns; a non-terminating rung loop is exactly such an overrun.
    // We prepend a per-loop iteration counter and assert it stays within a
    // bounded budget, turning a (trigger-gated) non-terminating Ladder Logic
    // Bomb into a reachable safety violation that incremental BMC detects,
    // while bounded legitimate loops within budget remain SAFE.
    symbol_exprt wd = resolve_("__wd" + std::to_string(wd_counter_++));
    const long long WDOG = 8;

    code_blockt instrumented;
    exprt inc(exprt::plus, wd.type());
    inc.copy_to_operands(wd, from_integer(BigInt(1), wd.type()));
    instrumented.copy_to_operands(code_assignt(wd, inc));
    instrumented.copy_to_operands(code_assertt(
      binary_relation_exprt(wd, "<=", from_integer(BigInt(WDOG), wd.type()))));
    for (const auto &op : body.operands())
      instrumented.copy_to_operands(static_cast<const codet &>(op));

    code_whilet loop;
    loop.cond() = cond;
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
