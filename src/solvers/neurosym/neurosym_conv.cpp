#include <solvers/neurosym/neurosym_conv.h>
#include <solvers/smtlib/oneshot_process.h>
#include <util/arith/mp_arith.h>
#include <util/message/message.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

static std::string prog_command(const optionst &options)
{
  std::string cmd = options.get_option("neurosym-prog");
  // Resolved through PATH, like the sibling bitwuzllob backend's "mallob"
  // default: NeuroSym ships a wrapper of this name alongside its native C++
  // solver. The original "python main.py %f" entry point still works when
  // named explicitly through --neurosym-prog.
  return cmd.empty() ? "neurosym-cpp-solve %f" : cmd;
}

static void skip_ws(const std::string &s, size_t &pos)
{
  while (pos < s.size() && std::isspace((unsigned char)s[pos]))
    pos++;
}

/* Skip one balanced-parens group starting at s[pos] == '(', leaving pos just
 * past its matching ')'. No-op (pos unchanged) if s[pos] is not '('. */
static void skip_paren_group(const std::string &s, size_t &pos)
{
  if (pos >= s.size() || s[pos] != '(')
    return;
  int depth = 0;
  do
  {
    if (s[pos] == '(')
      depth++;
    else if (s[pos] == ')')
      depth--;
    pos++;
  } while (pos < s.size() && depth > 0);
}

/* One bare token: runs until whitespace or a paren. */
static std::string read_token(const std::string &s, size_t &pos)
{
  size_t start = pos;
  while (pos < s.size() && !std::isspace((unsigned char)s[pos]) &&
         s[pos] != '(' && s[pos] != ')')
    pos++;
  return s.substr(start, pos - start);
}

/* Inverse of quote_smtlib_symbol() (smtlib_conv.cpp): ESBMC's mangled names
 * contain characters ('@', '$', ':', ...) illegal in an unquoted SMT-LIB2
 * symbol, so the serializer pipe-quotes every name it writes and escapes
 * the three characters a quoted symbol still cannot contain --
 * "/" -> "//", "\" -> "/b", "|" -> "/p" (applied in that order). NeuroSym's
 * own model output simply echoes back whatever name it read out of the
 * formula file, so it is in this *escaped* form -- but smt_ast::symname
 * (what get_bv()/l_get() are actually asked to look up) is always the raw,
 * unescaped name. Without reversing the escaping here, local_model's keys
 * and every lookup against them silently mismatch for any name containing
 * one of these three characters -- confirmed directly: a program's own
 * "guard" bookkeeping symbols are named with a literal '\' internally,
 * which is common enough (present in seemingly every ESBMC-generated
 * formula, not some rare corner case) that this was the single largest
 * remaining gap in running NeuroSym with no --neurosym-model-prog at all.
 * Must undo in the reverse of the encode order to stay unambiguous:
 * "/p" -> "|" first, then "/b" -> "\", then finally "//" -> "/". */
static std::string unescape_smtlib_name(const std::string &s)
{
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();)
  {
    if (s[i] == '/' && i + 1 < s.size())
    {
      if (s[i + 1] == 'p')
      {
        out += '|';
        i += 2;
        continue;
      }
      if (s[i + 1] == 'b')
      {
        out += '\\';
        i += 2;
        continue;
      }
      if (s[i + 1] == '/')
      {
        out += '/';
        i += 2;
        continue;
      }
    }
    out += s[i];
    i++;
  }
  return out;
}

/* Parse one "(define-fun NAME () SORT VALUE)" entry starting at s[pos],
 * leaving pos just past it. Returns false only when the text runs out
 * mid-entry, so the caller stops; an entry this cannot make sense of leaves
 * name or value empty and is skipped, which degrades that one variable to
 * the --neurosym-model-prog fallback. Split out of parse_model_block() to
 * keep it inside the repo's cyclomatic-complexity gate. */
static bool parse_define_fun(
  const std::string &s,
  size_t &pos,
  std::string &name,
  std::string &value)
{
  const size_t n = s.size();
  size_t entry_open = pos;
  pos++; // past '('
  skip_ws(s, pos);
  std::string keyword = read_token(s, pos);
  if (keyword != "define-fun")
  {
    // Not a form we understand (e.g. a comment leaked through as a list) --
    // skip this whole parenthesized entry and move on to the next.
    pos = entry_open;
    skip_paren_group(s, pos);
    return true;
  }

  skip_ws(s, pos);
  if (pos < n && s[pos] == '|')
  {
    size_t close = s.find('|', pos + 1);
    if (close == std::string::npos)
      return false;
    name = s.substr(pos + 1, close - pos - 1);
    pos = close + 1;
  }
  else
    name = read_token(s, pos);
  // NeuroSym echoes back the escaped/pipe-quoted form it read from the
  // formula file, not the raw name ESBMC's own AST holds (symname) --
  // see unescape_smtlib_name()'s comment. A no-op for a name that was
  // never escaped in the first place.
  name = unescape_smtlib_name(name);

  skip_ws(s, pos); // the empty parameter list "()"
  skip_paren_group(s, pos);

  skip_ws(s, pos); // the sort -- "(_ BitVec N)" or a bare "Int"/"Bool"
  if (pos < n && s[pos] == '(')
    skip_paren_group(s, pos);
  else
    read_token(s, pos);

  skip_ws(s, pos);
  if (pos < n && s[pos] == '(')
  {
    /* A parenthesized value, e.g. SMT-LIB2's "(- 5)" for a negative
     * numeral -- not currently interpreted; leave value empty so this one
     * variable falls back to --neurosym-model-prog instead of guessing. */
    skip_paren_group(s, pos);
  }
  else
    value = read_token(s, pos);

  return true;
}

void neurosym_convt::parse_model_block(const std::string &output)
{
  /* Purpose-built for exactly the format NeuroSym's own format_output()
   * emits (gansat/ns_solver.py, main.py):
   *   (model
   *     (define-fun NAME () SORT VALUE)
   *     ...
   *   )
   * SORT is "Int", "Bool", or "(_ BitVec N)"; VALUE is a bare decimal
   * numeral, a #x hex literal, or true/false. NAME may be pipe-quoted.
   * Anything this does not recognize is simply skipped for that one
   * variable -- it falls through to the --neurosym-model-prog fallback
   * exactly as if local parsing had not been attempted, so a NeuroSym
   * output format this cannot fully make sense of degrades to the old
   * behaviour rather than building a wrong counterexample. */
  size_t pos = output.find("(model");
  if (pos == std::string::npos)
    return;
  pos += 6;
  const size_t n = output.size();

  while (true)
  {
    skip_ws(output, pos);
    if (pos >= n || output[pos] == ')')
      break;
    if (output[pos] != '(')
      break; // unrecognized content where another define-fun was expected

    std::string name, value;
    if (!parse_define_fun(output, pos, name, value))
      break;
    if (!name.empty() && !value.empty())
      local_model[name] = value;

    skip_ws(output, pos);
    if (pos < n && output[pos] == ')')
      pos++; // close this define-fun
  }
}

/* Mirrors smtlib_convt's own interp_numeric() (smtlib_conv.cpp, file-local
 * there) closely enough to interpret local_model's raw value text the same
 * way a real solver's (get-value) response would be. */
static bool numeric_value(const std::string &v, bool is_signed, BigInt &out)
{
  if (v.size() > 2 && v[0] == '#' && v[1] == 'x')
  {
    out = string2integer(v.substr(2), 16);
    return true;
  }
  if (v.size() > 2 && v[0] == '#' && v[1] == 'b')
  {
    out = binary2integer(v.substr(2), is_signed);
    return true;
  }
  size_t digits_from = (!v.empty() && v[0] == '-') ? 1 : 0;
  if (
    v.size() > digits_from &&
    std::all_of(v.begin() + digits_from, v.end(), [](unsigned char c) {
      return std::isdigit(c);
    }))
  {
    out = string2integer(v);
    return true;
  }
  return false;
}

/* Mask/reinterpret `val` to exactly `width` bits, as an unsigned bit
 * pattern (i.e. the value a fixed-width register holding the low `width`
 * bits of val would show, 0 <= result < 2^width). Reuses
 * integer2binary()/binary2integer() (mp_arith.h) rather than hand-rolling
 * BigInt bit operations: integer2binary(val, width) already truncates to
 * the low `width` bits (two's-complement, so this is correct for a
 * negative val too), and binary2integer(..., false) reads that back
 * unsigned. */
static BigInt mask_to_width(const BigInt &val, std::size_t width)
{
  return binary2integer(integer2binary(val, width), false);
}

/* Reinterpret an unsigned `width`-bit pattern as a signed value (two's
 * complement): subtract 2^width if the top bit is set. */
static BigInt to_signed(const BigInt &unsigned_val, std::size_t width)
{
  BigInt top_bit = BigInt(1) << BigInt(width - 1);
  if (unsigned_val >= top_bit)
    return unsigned_val - (BigInt(1) << BigInt(width));
  return unsigned_val;
}

std::optional<BigInt>
neurosym_convt::local_lookup(const std::string &symname) const
{
  auto it = local_model.find(symname);
  if (it == local_model.end())
    return std::nullopt;
  BigInt m;
  if (!numeric_value(it->second, false, m))
    return std::nullopt;
  return m;
}

std::optional<BigInt> neurosym_convt::local_eval_array_at(
  smt_astt array_term,
  const BigInt &index) const
{
  const auto *ast = to_solver_smt_ast<smtlib_smt_ast>(array_term);

  switch (ast->kind)
  {
  case SMT_FUNC_STORE:
  {
    // args: (store base idx val)
    auto idx_val = local_eval_bv(ast->args[1]);
    if (!idx_val)
      return std::nullopt;
    if (*idx_val == index)
      return local_eval_bv(ast->args[2]);
    return local_eval_array_at(ast->args[0], index);
  }
  case SMT_FUNC_ITE:
  {
    // args: (ite cond then-array else-array)
    auto cond = local_eval_bool(ast->args[0]);
    if (!cond)
      return std::nullopt;
    return local_eval_array_at(*cond ? ast->args[1] : ast->args[2], index);
  }
  default:
    // A bare array symbol (or anything else this does not understand):
    // NeuroSym's model output has no whole-array representation to fall
    // back on locally, so this is an honest "don't know", not a bug.
    return std::nullopt;
  }
}

/* Bitwise ops over `width` bits: BigInt has no native bitwise operators, so
 * go via the binary strings mask_to_width() already relies on. */
template <typename F>
static BigInt
bitwise_bv(const BigInt &a, const BigInt &b, std::size_t width, F &&bit)
{
  std::string a_bits = integer2binary(a, width);
  std::string b_bits = integer2binary(b, width);
  std::string out(width, '0');
  for (std::size_t i = 0; i < width; i++)
    out[i] = bit(a_bits[i] == '1', b_bits[i] == '1') ? '1' : '0';
  return binary2integer(out, false);
}

/* The binary bit-vector operators, split out of local_eval_bv() to keep that
 * function inside the repo's cyclomatic-complexity gate. Operands and result
 * are unsigned `width`-bit patterns. */
static std::optional<BigInt> eval_bv_binop(
  smt_func_kind kind,
  const BigInt &lhs,
  const BigInt &rhs,
  std::size_t width)
{
  switch (kind)
  {
  case SMT_FUNC_ADD:
  case SMT_FUNC_BVADD:
    return mask_to_width(lhs + rhs, width);
  case SMT_FUNC_SUB:
  case SMT_FUNC_BVSUB:
    return mask_to_width(lhs - rhs, width);
  case SMT_FUNC_MUL:
  case SMT_FUNC_BVMUL:
    return mask_to_width(lhs * rhs, width);
  case SMT_FUNC_BVUDIV:
    if (rhs == BigInt(0))
      return mask_to_width(
        (BigInt(1) << BigInt(width)) - BigInt(1),
        width); // SMT-LIB2 udiv-by-0
    return mask_to_width(lhs / rhs, width);
  case SMT_FUNC_BVSDIV:
  {
    if (rhs == BigInt(0))
      return mask_to_width(
        to_signed(lhs, width) >= BigInt(0)
          ? (BigInt(1) << BigInt(width)) - BigInt(1)
          : BigInt(1),
        width);
    BigInt q = to_signed(lhs, width) / to_signed(rhs, width);
    return mask_to_width(q, width);
  }
  case SMT_FUNC_BVUMOD:
    if (rhs == BigInt(0))
      return mask_to_width(lhs, width); // SMT-LIB2 urem-by-0
    return mask_to_width(lhs % rhs, width);
  case SMT_FUNC_BVSMOD:
  {
    if (rhs == BigInt(0))
      return mask_to_width(lhs, width);
    BigInt r = to_signed(lhs, width) % to_signed(rhs, width);
    return mask_to_width(r, width);
  }
  /* A shift by >= the operand width shifts every bit out: zero for
   * bvshl/bvlshr, the replicated sign bit for bvashr (SMT-LIB2
   * FixedSizeBitVectors). The guard is load-bearing, not a nicety:
   * mp_arith's operator<< / operator>> both go through power(2, n),
   * which loops n BigInt multiplications, so an unguarded shift amount
   * taken from the model (any value up to 2^width - 1) does not
   * terminate in practice. */
  case SMT_FUNC_BVSHL:
  case SMT_FUNC_SHL:
    if (rhs >= BigInt(width))
      return BigInt(0);
    return mask_to_width(lhs << rhs, width);
  case SMT_FUNC_BVLSHR:
    if (rhs >= BigInt(width))
      return BigInt(0);
    return mask_to_width(lhs >> rhs, width);
  case SMT_FUNC_BVASHR:
  {
    if (rhs >= BigInt(width))
      return to_signed(lhs, width) < BigInt(0)
               ? mask_to_width(BigInt(-1), width)
               : BigInt(0);
    BigInt shifted = to_signed(lhs, width) >> rhs;
    return mask_to_width(shifted, width);
  }
  case SMT_FUNC_BVAND:
    return bitwise_bv(lhs, rhs, width, [](bool a, bool b) { return a && b; });
  case SMT_FUNC_BVOR:
    return bitwise_bv(lhs, rhs, width, [](bool a, bool b) { return a || b; });
  case SMT_FUNC_BVXOR:
    return bitwise_bv(lhs, rhs, width, [](bool a, bool b) { return a != b; });
  default:
    return std::nullopt; // unreachable given the outer switch
  }
}

std::optional<BigInt> neurosym_convt::local_eval_bv(smt_astt a) const
{
  const auto *ast = to_solver_smt_ast<smtlib_smt_ast>(a);
  const std::size_t width = a->sort->get_data_width();

  switch (ast->kind)
  {
  case SMT_FUNC_INT:
  case SMT_FUNC_BVINT:
    return mask_to_width(ast->intval, width);

  case SMT_FUNC_SYMBOL:
  {
    /* Mask like every other arm: NeuroSym may print a value in any form
     * numeric_value() accepts, including a negative or out-of-range
     * decimal. Returning it verbatim leaks a value that is not a canonical
     * unsigned width-bit pattern, which to_signed() then misreads and which
     * slips past the shift guards below into power(2, negative). */
    auto v = local_lookup(ast->symname);
    if (!v)
      return std::nullopt;
    return mask_to_width(*v, width);
  }

  case SMT_FUNC_SELECT:
  {
    // Unreachable while arrays are flattened; see local_eval_array_at().
    // args: (select array idx)
    auto idx_val = local_eval_bv(ast->args[1]);
    if (!idx_val)
      return std::nullopt;
    return local_eval_array_at(ast->args[0], *idx_val);
  }

  case SMT_FUNC_EXTRACT:
  {
    auto src = local_eval_bv(ast->args[0]);
    if (!src)
      return std::nullopt;
    // extract_high/extract_low are inclusive bit indices into the source;
    // shift the low bit to position 0, then mask to the extracted width.
    BigInt shifted = *src >> BigInt(ast->extract_low);
    return mask_to_width(shifted, width);
  }

  case SMT_FUNC_CONCAT:
  {
    // args: (concat high low) -- SMT-LIB2 puts the more-significant operand
    // first; the result width is the sum of both operand widths.
    auto hi = local_eval_bv(ast->args[0]);
    auto lo = local_eval_bv(ast->args[1]);
    if (!hi || !lo)
      return std::nullopt;
    std::size_t lo_width = ast->args[1]->sort->get_data_width();
    return mask_to_width((*hi << BigInt(lo_width)) + *lo, width);
  }

  case SMT_FUNC_ITE:
  {
    auto cond = local_eval_bool(ast->args[0]);
    if (!cond)
      return std::nullopt;
    return local_eval_bv(*cond ? ast->args[1] : ast->args[2]);
  }

  case SMT_FUNC_BVNEG:
  case SMT_FUNC_NEG:
  {
    auto v = local_eval_bv(ast->args[0]);
    if (!v)
      return std::nullopt;
    return mask_to_width(-*v, width);
  }

  case SMT_FUNC_BVNOT:
  {
    auto v = local_eval_bv(ast->args[0]);
    if (!v)
      return std::nullopt;
    // Bitwise not over `width` bits: (2^width - 1) - v.
    BigInt all_ones = (BigInt(1) << BigInt(width)) - BigInt(1);
    return mask_to_width(all_ones - *v, width);
  }

  case SMT_FUNC_ADD:
  case SMT_FUNC_BVADD:
  case SMT_FUNC_SUB:
  case SMT_FUNC_BVSUB:
  case SMT_FUNC_MUL:
  case SMT_FUNC_BVMUL:
  case SMT_FUNC_BVUDIV:
  case SMT_FUNC_BVSDIV:
  case SMT_FUNC_BVUMOD:
  case SMT_FUNC_BVSMOD:
  case SMT_FUNC_BVSHL:
  case SMT_FUNC_SHL:
  case SMT_FUNC_BVLSHR:
  case SMT_FUNC_BVASHR:
  case SMT_FUNC_BVAND:
  case SMT_FUNC_BVOR:
  case SMT_FUNC_BVXOR:
  {
    auto lhs = local_eval_bv(ast->args[0]);
    auto rhs = local_eval_bv(ast->args[1]);
    if (!lhs || !rhs)
      return std::nullopt;
    return eval_bv_binop(ast->kind, *lhs, *rhs, width);
  }

  default:
    // Floating-point ops, uninterpreted functions, real/int conversions:
    // NeuroSym has no native support for any of these (see the class
    // comment), so a query landing here falls back to
    // --neurosym-model-prog, same as before this evaluator existed.
    return std::nullopt;
  }
}

/* The bit-vector comparisons, split out of local_eval_bool() to keep that
 * function inside the repo's cyclomatic-complexity gate. Operands are
 * unsigned `width`-bit patterns; signedness comes from the operator. */
static std::optional<bool> eval_bv_compare(
  smt_func_kind kind,
  const BigInt &lhs,
  const BigInt &rhs,
  std::size_t width)
{
  bool is_signed_cmp = kind == SMT_FUNC_LT || kind == SMT_FUNC_GT ||
                       kind == SMT_FUNC_LTE || kind == SMT_FUNC_GTE ||
                       kind == SMT_FUNC_BVSLT || kind == SMT_FUNC_BVSGT ||
                       kind == SMT_FUNC_BVSLTE || kind == SMT_FUNC_BVSGTE;
  BigInt l = is_signed_cmp ? to_signed(lhs, width) : lhs;
  BigInt r = is_signed_cmp ? to_signed(rhs, width) : rhs;
  switch (kind)
  {
  case SMT_FUNC_LT:
  case SMT_FUNC_BVSLT:
  case SMT_FUNC_BVULT:
    return l < r;
  case SMT_FUNC_GT:
  case SMT_FUNC_BVSGT:
  case SMT_FUNC_BVUGT:
    return l > r;
  case SMT_FUNC_LTE:
  case SMT_FUNC_BVSLTE:
  case SMT_FUNC_BVULTE:
    return l <= r;
  case SMT_FUNC_GTE:
  case SMT_FUNC_BVSGTE:
  case SMT_FUNC_BVUGTE:
    return l >= r;
  default:
    return std::nullopt;
  }
}

std::optional<bool>
neurosym_convt::local_lookup_bool(const std::string &symname) const
{
  /* local_lookup() only understands numeric text; a genuinely Bool-sorted
   * model entry is textual ("true"/"false"), as NeuroSym's own define-fun
   * output and parse_model_block() emit it. Handle that form directly
   * rather than failing through local_lookup()'s numeric parse and forcing
   * every boolean-guarded trace out to the external fallback solver. */
  auto it = local_model.find(symname);
  if (it == local_model.end())
    return std::nullopt;
  if (it->second == "true")
    return true;
  if (it->second == "false")
    return false;
  auto v = local_lookup(symname);
  if (!v)
    return std::nullopt;
  return *v != BigInt(0);
}

std::optional<bool> neurosym_convt::eval_bool_fold(
  smt_func_kind kind,
  const smtlib_smt_ast *ast) const
{
  // ESBMC commonly ANDs/ORs many guard literals together in one node
  // (not necessarily just two), e.g. combining a whole path condition --
  // fold left over however many args are actually there.
  if (ast->args.empty())
    return std::nullopt;
  auto acc = local_eval_bool(ast->args[0]);
  if (!acc)
    return std::nullopt;
  bool result = *acc;
  for (std::size_t i = 1; i < ast->args.size(); i++)
  {
    auto v = local_eval_bool(ast->args[i]);
    if (!v)
      return std::nullopt;
    switch (kind)
    {
    case SMT_FUNC_AND:
      result = result && *v;
      break;
    case SMT_FUNC_OR:
      result = result || *v;
      break;
    case SMT_FUNC_XOR:
      result = result != *v;
      break;
    default:
      return std::nullopt;
    }
  }
  return result;
  return result;
}

std::optional<bool>
neurosym_convt::eval_bool_eq(const smtlib_smt_ast *ast) const
{
  /* The operand sort decides whether to compare as bit-vectors or as
   * booleans; array/tuple equality is not handled (rare for a counterexample
   * query, and it falls back). */
  if (ast->args[0]->sort->id == SMT_SORT_BOOL)
  {
    auto lhs = local_eval_bool(ast->args[0]);
    auto rhs = local_eval_bool(ast->args[1]);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs == *rhs;
  }
  auto lhs = local_eval_bv(ast->args[0]);
  auto rhs = local_eval_bv(ast->args[1]);
  if (!lhs || !rhs)
    return std::nullopt;
  return *lhs == *rhs;
}

std::optional<bool> neurosym_convt::local_eval_bool(smt_astt a) const
{
  const auto *ast = to_solver_smt_ast<smtlib_smt_ast>(a);

  switch (ast->kind)
  {
  case SMT_FUNC_BOOL:
    return ast->boolval;

  case SMT_FUNC_SYMBOL:
    return local_lookup_bool(ast->symname);

  case SMT_FUNC_NOT:
  {
    auto v = local_eval_bool(ast->args[0]);
    if (!v)
      return std::nullopt;
    return !*v;
  }

  case SMT_FUNC_AND:
  case SMT_FUNC_OR:
  case SMT_FUNC_XOR:
    return eval_bool_fold(ast->kind, ast);

  case SMT_FUNC_IMPLIES:
  {
    // Binary by SMT-LIB2 definition, unlike AND/OR/XOR above.
    if (ast->args.size() != 2)
      return std::nullopt;
    auto lhs = local_eval_bool(ast->args[0]);
    auto rhs = local_eval_bool(ast->args[1]);
    if (!lhs || !rhs)
      return std::nullopt;
    return !*lhs || *rhs;
  }

  case SMT_FUNC_ITE:
  {
    auto cond = local_eval_bool(ast->args[0]);
    if (!cond)
      return std::nullopt;
    return local_eval_bool(*cond ? ast->args[1] : ast->args[2]);
  }

  case SMT_FUNC_EQ:
  case SMT_FUNC_NOTEQ:
  {
    auto eq = eval_bool_eq(ast);
    if (!eq)
      return std::nullopt;
    return ast->kind == SMT_FUNC_EQ ? *eq : !*eq;
  }

  case SMT_FUNC_LT:
  case SMT_FUNC_GT:
  case SMT_FUNC_LTE:
  case SMT_FUNC_GTE:
  case SMT_FUNC_BVSLT:
  case SMT_FUNC_BVULT:
  case SMT_FUNC_BVSGT:
  case SMT_FUNC_BVUGT:
  case SMT_FUNC_BVSLTE:
  case SMT_FUNC_BVULTE:
  case SMT_FUNC_BVSGTE:
  case SMT_FUNC_BVUGTE:
  {
    auto lhs = local_eval_bv(ast->args[0]);
    auto rhs = local_eval_bv(ast->args[1]);
    if (!lhs || !rhs)
      return std::nullopt;
    return eval_bv_compare(
      ast->kind, *lhs, *rhs, ast->args[0]->sort->get_data_width());
  }

  default:
    return std::nullopt;
  }
}

smt_solver_baset *create_new_neurosym_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api [[maybe_unused]],
  array_iface **array_api [[maybe_unused]],
  fp_convt **fp_api [[maybe_unused]])
{
  /* NeuroSym solves a single formula per invocation; strategies that reuse
   * one persistent solver context across repeated or incremental checks
   * cannot be served by it. --multi-property is NOT in this list: without
   * --smt-during-symex, bmct::multi_property_check() (bmc.cpp) allocates a
   * fresh create_solver() instance per claim rather than reusing one across
   * claims, which is exactly NeuroSym's one-shot-per-invocation model — so
   * --multi-property (and the coverage modes built on it, e.g.
   * --branch-coverage) work correctly, just at the cost of one NeuroSym
   * subprocess per claim. Only --smt-during-symex, which explicitly shares
   * one persistent solver across every claim, is genuinely incompatible. */
  static const char *incompatible[] = {
    "incremental-bmc",
    "falsification",
    "k-induction",
    "k-induction-parallel",
    "termination",
    "smt-during-symex",
    "parallel-solving"};
  for (const char *opt : incompatible)
    if (options.get_bool_option(opt))
    {
      log_error(
        "the neurosym backend runs NeuroSym in one-shot batch mode and does "
        "not support --{}; use a linked solver (e.g. --bitwuzla) for "
        "incremental strategies",
        opt);
      abort();
    }

  /* NeuroSym is QF_BV-only: leaving the tuple/array/fp interfaces unset makes
   * create_solver() install the flatteners that lower structs, arrays and
   * floating-point to pure bit-vectors before they reach the serializer. */
  return new neurosym_convt(ns, options);
}

neurosym_convt::neurosym_convt(const namespacet &ns, const optionst &options)
  : neurosym_convt(
      ns,
      options,
      oneshot_process::choose_formula_path(options, "neurosym"))
{
}

neurosym_convt::neurosym_convt(
  const namespacet &ns,
  const optionst &options,
  const std::string &_formula_path)
  : smtlib_convt(
      ns,
      options,
      oneshot_process::model_prog(options, "neurosym"),
      _formula_path,
      "QF_BV"),
    formula_path(_formula_path)
{
}

neurosym_convt::~neurosym_convt()
{
  if (oneshot_process::uses_temp_formula(options))
    remove(formula_path.c_str());
}

std::string neurosym_convt::dump_smt()
{
  /* Under --smt-formula-only no solve follows; complete the dump with the
   * (check-sat) like the base class. Under --smt-formula-too our dec_solve()
   * emits the (check-sat) itself: appending one here as well would hand
   * NeuroSym a formula containing two. The base implementation also reports
   * the destination from the --output option, which this backend redirects
   * to the formula file. */
  if (options.get_bool_option("smt-formula-only"))
    return smtlib_convt::dump_smt();
  log_status("SMT formula written to {}", formula_path);
  return "SMT formula dumped successfully";
}

smt_resultt neurosym_convt::dec_solve()
{
  if (solved)
  {
    log_error(
      "the neurosym backend supports a single (check-sat) query per run; "
      "incremental strategies are not supported");
    abort();
  }
  solved = true;

  pre_solve();

  /* The (check-sat) goes to both sinks: the formula file for NeuroSym, and
   * the local model solver's pipe (if configured), which starts solving in
   * parallel and only gets waited for when a model is actually needed. The
   * model solver only produces counterexamples, so if it has died (e.g. it
   * failed to start), disable it and let NeuroSym decide: an unsat proof
   * needs no model, and a sat result reports the missing-model error below
   * rather than crashing on an uncaught exception. */
  try
  {
    emit_check_sat();
  }
  catch (const external_process_died &)
  {
    log_warning(
      "neurosym: the local model solver '{}' terminated unexpectedly (e.g. "
      "it failed to start); continuing without counterexample support",
      options.get_option("neurosym-model-prog"));
    emit_proc.terminate();
    flush(); // complete the formula file for NeuroSym now that the pipe is gone
  }

  std::string captured_output;
  smt_resultt res = oneshot_process::run_solver(
    prog_command(options), formula_path, "neurosym", &captured_output);
  if (res != P_SATISFIABLE)
  {
    /* No model will be read; stop the local solver we fed in parallel rather
     * than let it keep solving until this object is destroyed. */
    emit_proc.terminate();
    return res;
  }

  /* NeuroSym's own batch stdout already carries a sort-correct model on a
   * sat verdict (see the class comment): parse it directly rather than
   * unconditionally waiting on the local model solver's answer here. When it
   * covers everything the trace ends up asking for -- the common case, since
   * NeuroSym's model output normally lists every free variable in the
   * formula -- --neurosym-model-prog's solve is never waited for at all,
   * however long it takes on this formula. A variable that is not in
   * local_model (parsing failure, or a value form local parsing does not
   * understand) still falls back to it, lazily, the first time get_bv() /
   * l_get() below actually needs it. */
  parse_model_block(captured_output);
  if (!local_model.empty())
    return P_SATISFIABLE;

  /* Local parsing found nothing usable (NeuroSym's output did not match the
   * expected model format, or this particular formula has no free
   * variables) -- fall back to the original --neurosym-model-prog path in
   * full, right away, exactly as before this change. */
  return ensure_model_prog_ready() || options.get_bool_option("result-only")
           ? P_SATISFIABLE
           : P_ERROR;
}

bool neurosym_convt::ensure_model_prog_ready()
{
  if (model_prog_response_read)
    return static_cast<bool>(emit_proc);
  model_prog_response_read = true;

  /* A satisfiable formula with nothing usable in local_model needs a live
   * model solver to turn into a counterexample. It is absent either because
   * the model solver died earlier (a command was given) or was never
   * configured. Under --result-only no counterexample is ever built
   * (bmc.cpp skips trace construction) and get_bv()/l_get() are normally
   * never even called -- but dec_solve() itself still calls this eagerly as
   * its own fallback when local_model comes up empty, regardless of
   * --result-only, so that path still needs handling here explicitly: stay
   * silent and let the (never-built) trace go on rather than erroring over
   * a model nothing will read. */
  if (!emit_proc)
  {
    if (options.get_bool_option("result-only"))
      return false;
    if (options.get_option("neurosym-model-prog").empty())
      log_error(
        "neurosym: formula is satisfiable, but building the counterexample "
        "requires a local interactive SMT-LIB2 solver; re-run with "
        "--neurosym-model-prog <cmd> (e.g. \"z3 -in\") or with "
        "--result-only");
    else
      log_error(
        "neurosym: the local model solver is unavailable; cannot build a "
        "counterexample");
    return false;
  }

  smt_resultt model_res;
  try
  {
    model_res = read_check_sat_response();
  }
  catch (const external_process_died &)
  {
    log_error(
      "neurosym: the local model solver is unavailable; cannot build a "
      "counterexample");
    return false;
  }
  if (model_res != P_SATISFIABLE)
  {
    log_error(
      "neurosym: NeuroSym reported sat but the local model solver did not; "
      "refusing to build a counterexample from a diverging model");
    abort();
  }
  return true;
}

tvt neurosym_convt::get_bool(smt_astt a)
{
  return l_get(a);
}

tvt neurosym_convt::l_get(smt_astt a)
{
  const std::string &symname = to_solver_smt_ast<smtlib_smt_ast>(a)->symname;
  if (!symname.empty())
  {
    auto it = local_model.find(symname);
    if (it != local_model.end())
    {
      const std::string &v = it->second;
      if (v == "true")
        return tvt(true);
      if (v == "false")
        return tvt(false);
      BigInt m;
      if (numeric_value(v, false, m))
        return tvt(m != 0);
      // Fall through: an entry exists but this parser could not make sense
      // of its value (e.g. the "(- N)" form) -- treat it the same as a miss.
    }
  }

  /* Either a composite expression (empty symname) or a leaf miss: try
   * evaluating it locally from local_model before paying for the live
   * solver at all. */
  if (auto v = local_eval_bool(a))
    return tvt(*v);

  ensure_model_prog_ready();
  /* Reaching here without a model solver means --result-only: any other
   * configuration made dec_solve() return P_ERROR. error_trace() bails out
   * immediately in that mode (bmc.cpp:327), so no counterexample is built
   * from this value; the caller is the property-identification pass, which
   * reports the claim as not checked. Falling through to
   * smtlib_convt::l_get() would instead send a (get-value) to a solver that
   * was never started and block forever reading its response. */
  if (!emit_proc)
    return tvt(tvt::TV_UNKNOWN);
  return smtlib_convt::l_get(a);
}

BigInt neurosym_convt::get_bv(smt_astt a, bool is_signed)
{
  const std::string &symname = to_solver_smt_ast<smtlib_smt_ast>(a)->symname;
  if (!symname.empty())
  {
    auto it = local_model.find(symname);
    if (it != local_model.end())
    {
      BigInt m;
      if (numeric_value(it->second, is_signed, m))
        return m;
    }
  }

  /* Composite expression (array select, pointer-offset arithmetic, ...) or
   * a leaf miss: evaluate it locally before falling back to the live
   * solver. local_eval_bv() always returns an unsigned bit pattern;
   * reinterpret it as signed here if the caller asked for that, matching
   * how smtlib_convt::get_bv()/interp_numeric() already handle is_signed
   * for a plain (get-value) response. */
  if (auto v = local_eval_bv(a))
  {
    if (!is_signed)
      return *v;
    return to_signed(*v, a->sort->get_data_width());
  }

  ensure_model_prog_ready();
  // See the matching comment in l_get(): without a usable emit_proc this
  // would otherwise block forever reading a (get-value) response from a
  // solver that was never started.
  if (!emit_proc)
    return BigInt(0);
  return smtlib_convt::get_bv(a, is_signed);
}

const std::string neurosym_convt::solver_text()
{
  // Just the friendly name -- the full invocation (path + every flag +
  // the temp formula path) is verbose noise at normal verbosity; it is
  // still available via log_debug("solver", ...) in oneshot_process.cpp
  // for anyone who needs the exact command.
  return "NeuroSym";
}
