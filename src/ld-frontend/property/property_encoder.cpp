#include <ld-frontend/property/property_encoder.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/symbol.h>
#include <cctype>
#include <stdexcept>

property_encoder::property_encoder(
  contextt &context,
  const std::string &source_file)
  : context_(context), source_file_(source_file)
{
}

static std::string ld_name(const std::string &var)
{
  return "ld::" + var;
}

symbol_exprt
property_encoder::var_expr(const std::string &name, const LdProperty &p) const
{
  const symbolt *sym = context_.find_symbol(ld_name(name));
  if (!sym)
    throw std::runtime_error(
      "property '" + p.id + "': undeclared variable '" + name + "'");
  return symbol_exprt(ld_name(name), sym->get_type());
}

// Find the leftmost occurrence of 'op' at parenthesis depth 0.
static size_t find_outermost_op(const std::string &s, const std::string &op)
{
  int depth = 0;
  for (size_t i = 0; i + op.size() <= s.size(); ++i)
  {
    if (s[i] == '(')
      ++depth;
    else if (s[i] == ')')
      --depth;
    else if (depth == 0 && s.compare(i, op.size(), op) == 0)
      return i;
  }
  return std::string::npos;
}

// Minimal boolean expression parser: var, !var, A && B, A || B, (expr).
exprt property_encoder::parse_bool_expr(
  const std::string &expr_str,
  const LdProperty &p) const
{
  // Trim whitespace.
  std::string s = expr_str;
  while (!s.empty() && isspace(static_cast<unsigned char>(s.front())))
    s.erase(s.begin());
  while (!s.empty() && isspace(static_cast<unsigned char>(s.back())))
    s.pop_back();

  if (s.empty())
    throw std::runtime_error(
      "property '" + p.id + "': empty boolean expression");

  // Strip outer parentheses when the whole expression is wrapped.
  if (s.front() == '(' && s.back() == ')')
  {
    // Verify the outer parens actually match (not '(A) || (B)').
    int depth = 0;
    bool outer_matched = true;
    for (size_t i = 0; i < s.size() - 1; ++i)
    {
      if (s[i] == '(')
        ++depth;
      else if (s[i] == ')')
      {
        --depth;
        if (depth == 0)
        {
          outer_matched = false;
          break;
        }
      }
    }
    if (outer_matched)
      return parse_bool_expr(s.substr(1, s.size() - 2), p);
  }

  // Scan for outermost || first (lowest precedence).
  size_t or_pos = find_outermost_op(s, "||");
  if (or_pos != std::string::npos)
  {
    exprt lhs = parse_bool_expr(s.substr(0, or_pos), p);
    exprt rhs = parse_bool_expr(s.substr(or_pos + 2), p);
    exprt result(exprt::i_or, typet("bool"));
    result.copy_to_operands(lhs, rhs);
    return result;
  }

  // Then &&.
  size_t and_pos = find_outermost_op(s, "&&");
  if (and_pos != std::string::npos)
    return and_exprt(
      parse_bool_expr(s.substr(0, and_pos), p),
      parse_bool_expr(s.substr(and_pos + 2), p));

  // Negation.
  if (s.front() == '!')
    return not_exprt(parse_bool_expr(s.substr(1), p));

  return var_expr(s, p);
}

code_assertt
property_encoder::make_assert(const exprt &cond, const LdProperty &p) const
{
  code_assertt asrt(cond);
  locationt loc;
  loc.set_file(source_file_);
  loc.property(p.id);
  loc.comment(p.description.empty() ? p.id : p.description);
  asrt.location() = loc;
  return asrt;
}

// mutual_exclusion: assert !(A && B && ...)
code_blockt property_encoder::encode_mutual_exclusion(const LdProperty &p)
{
  // Caller (yaml_property_parser) guarantees >= 2 variables.
  exprt conjunction = var_expr(p.variables[0], p);
  for (size_t i = 1; i < p.variables.size(); ++i)
    conjunction = and_exprt(conjunction, var_expr(p.variables[i], p));

  code_blockt blk;
  blk.copy_to_operands(make_assert(not_exprt(conjunction), p));
  return blk;
}

// invariant: assert expr
code_blockt property_encoder::encode_invariant(const LdProperty &p)
{
  code_blockt blk;
  blk.copy_to_operands(make_assert(parse_bool_expr(p.expression, p), p));
  return blk;
}

// absence: assert !expr
code_blockt property_encoder::encode_absence(const LdProperty &p)
{
  code_blockt blk;
  blk.copy_to_operands(
    make_assert(not_exprt(parse_bool_expr(p.expression, p)), p));
  return blk;
}

// reachability: if (guard) assert(false) — ESBMC BMC finds the path
code_blockt property_encoder::encode_reachability(const LdProperty &p)
{
  code_blockt blk;
  code_ifthenelset ite;
  ite.cond() = parse_bool_expr(p.expression, p);
  ite.then_case() = make_assert(false_exprt(), p);
  blk.copy_to_operands(ite);
  return blk;
}

// response: auxiliary scan counter; assert trigger => response within max_scans
code_blockt property_encoder::encode_response(const LdProperty &p)
{
  std::string ctr_name = "ld::__resp_ctr_" + p.id;
  if (!context_.find_symbol(ctr_name))
  {
    symbolt ctr_sym;
    ctr_sym.id = ctr_name;
    ctr_sym.name = "__resp_ctr_" + p.id;
    ctr_sym.module = "ld";
    ctr_sym.mode = "LD";
    ctr_sym.set_type(int_type());
    ctr_sym.set_value(from_integer(BigInt(0), int_type()));
    ctr_sym.static_lifetime = true;
    ctr_sym.lvalue = true;
    locationt loc;
    loc.set_file(source_file_);
    ctr_sym.location = loc;
    context_.move_symbol_to_context(ctr_sym);
  }

  symbol_exprt ctr(ctr_name, int_type());
  symbol_exprt trigger = var_expr(p.trigger, p);
  symbol_exprt response = var_expr(p.response_var, p);
  exprt limit = from_integer(BigInt(p.max_scans), int_type());
  exprt one = gen_one(int_type());
  exprt zero = gen_zero(int_type());

  code_ifthenelset ctr_step;
  ctr_step.cond() = and_exprt(trigger, not_exprt(response));
  ctr_step.then_case() = code_assignt(ctr, plus_exprt(ctr, one));
  ctr_step.else_case() = code_assignt(ctr, zero);

  code_blockt blk;
  blk.copy_to_operands(ctr_step);
  blk.copy_to_operands(make_assert(binary_relation_exprt(ctr, "<=", limit), p));
  return blk;
}

code_blockt property_encoder::encode(const std::vector<LdProperty> &props)
{
  code_blockt blk;
  for (const auto &p : props)
  {
    code_blockt sub;
    switch (p.kind)
    {
    case PropertyKind::mutual_exclusion:
      sub = encode_mutual_exclusion(p);
      break;
    case PropertyKind::invariant:
      sub = encode_invariant(p);
      break;
    case PropertyKind::absence:
      sub = encode_absence(p);
      break;
    case PropertyKind::reachability:
      sub = encode_reachability(p);
      break;
    case PropertyKind::response:
      sub = encode_response(p);
      break;
    }
    for (auto &op : sub.operands())
      blk.copy_to_operands(op);
  }
  return blk;
}
