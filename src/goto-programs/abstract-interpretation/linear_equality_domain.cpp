/// \file
/// Linear Equality Domain

#include <goto-programs/abstract-interpretation/linear_equality_domain.h>

#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/message.h>

#include <sstream>

AffineExpr AffineExpr::operator+(const AffineExpr &rhs) const
{
  AffineExpr result;
  result.constant = constant + rhs.constant;
  result.coeffs = coeffs;
  for (const auto &kv : rhs.coeffs)
  {
    auto &slot = result.coeffs[kv.first];
    slot += kv.second;
    if (slot == 0)
      // cancel out if coefficients sum to zero
      result.coeffs.erase(kv.first);
  }
  return result;
}

AffineExpr AffineExpr::operator-(const AffineExpr &rhs) const
{
  return *this + (rhs * BigInt(-1));
}

AffineExpr AffineExpr::operator*(const BigInt &scalar) const
{
  if (scalar == 0)
    return AffineExpr(BigInt(0));
  AffineExpr result;
  result.constant = constant * scalar;
  for (const auto &kv : coeffs)
    result.coeffs[kv.first] = kv.second * scalar;
  return result;
}

std::string AffineExpr::to_string() const
{
  std::ostringstream oss;
  if (coeffs.empty())
  {
    oss << constant;
    return oss.str();
  }
  bool first = true;
  for (const auto &kv : coeffs)
  {
    if (!first)
      oss << " + ";
    first = false;
    if (kv.second != 1)
      oss << kv.second << "*";
    oss << kv.first;
  }
  if (constant != 0)
    oss << " + " << constant;
  return oss.str();
}

void linear_equality_domaint::copy_if_needed()
{
  if (equations.use_count() > 1)
    equations = std::make_shared<equation_map>(*equations);
}

AffineExpr linear_equality_domaint::substitute(
  const AffineExpr &target,
  const irep_idt &var,
  const AffineExpr &expr)
{
  auto it = target.coeffs.find(var);
  if (it == target.coeffs.end())
    return target; // var does not appear — nothing to do

  // Remove the var term, then add coeff * expr in its place.
  const BigInt coeff = it->second;
  AffineExpr result;
  result.constant = target.constant;
  for (const auto &kv : target.coeffs)
    if (kv.first != var)
      result.coeffs[kv.first] = kv.second;
  return result + (expr * coeff);
}

void linear_equality_domaint::assign_affine(
  const irep_idt &var,
  const AffineExpr &rhs)
{
  copy_if_needed();

  auto var_coeff_it = rhs.coeffs.find(var);

  if (var_coeff_it == rhs.coeffs.end())
  {
    // Case A: x = f(others) — substitute x_old -> rhs in every other equation.
    for (auto &kv : *equations)
      if (kv.first != var)
        kv.second = substitute(kv.second, var, rhs);
    (*equations)[var] = rhs;
  }
  else if (var_coeff_it->second == 1)
  {
    // Case B: x = x + delta.
    // Invert to x_old = x_new - delta, then substitute into other equations.
    AffineExpr rhs_no_x;
    rhs_no_x.constant = rhs.constant;
    for (const auto &kv : rhs.coeffs)
      if (kv.first != var)
        rhs_no_x.coeffs[kv.first] = kv.second;

    AffineExpr inverse_expr = AffineExpr(var, BigInt(1)) - rhs_no_x;

    // Save x's old equation before modifying the map.
    AffineExpr old_eq;
    bool had_equation = false;
    auto cur_it = equations->find(var);
    if (cur_it != equations->end())
    {
      old_eq = cur_it->second;
      had_equation = true;
    }

    for (auto &kv : *equations)
      if (kv.first != var)
        kv.second = substitute(kv.second, var, inverse_expr);

    if (had_equation)
      // x_new = x_old + delta
      (*equations)[var] = old_eq + rhs_no_x;
    else
      // x was free; still free after increment
      equations->erase(var);
  }
  else
  {
    // Case C: coefficient != 1 — inversion is not integer-linear; forget x.
    forget(var);
  }
}

void linear_equality_domaint::forget(const irep_idt &var)
{
  copy_if_needed();
  equations->erase(var);
  // Also drop any equation whose RHS mentions var — those are no longer sound.
  for (auto it = equations->begin(); it != equations->end();)
  {
    if (it->second.coeffs.count(var))
      it = equations->erase(it);
    else
      ++it;
  }
}

bool linear_equality_domaint::try_as_affine(
  const expr2tc &expr,
  AffineExpr &out) const
{
  if (is_nil_expr(expr))
    return false;

  if (is_constant_int2t(expr))
  {
    out = AffineExpr(to_constant_int2t(expr).value);
    return true;
  }

  if (is_symbol2t(expr))
  {
    if (
      !is_signedbv_type(expr->type) && !is_unsignedbv_type(expr->type) &&
      !is_bool_type(expr->type))
      return false;
    // Represent as a coefficient-1 linear term; do not substitute through the
    // equation map — keeping symbols raw lets merge preserve relative equalities
    // such as "y == x" across loop iterations.
    out = AffineExpr(to_symbol2t(expr).thename, BigInt(1));
    return true;
  }

  if (is_typecast2t(expr))
  {
    const auto &tc = to_typecast2t(expr);
    if (
      !is_signedbv_type(tc.type) && !is_unsignedbv_type(tc.type) &&
      !is_bool_type(tc.type))
      return false;
    return try_as_affine(tc.from, out);
  }

  if (is_neg2t(expr))
  {
    AffineExpr inner;
    if (!try_as_affine(to_neg2t(expr).value, inner))
      return false;
    out = inner * BigInt(-1);
    return true;
  }

  if (is_add2t(expr))
  {
    const auto &a = to_add2t(expr);
    AffineExpr lhs, rhs;
    if (!try_as_affine(a.side_1, lhs) || !try_as_affine(a.side_2, rhs))
      return false;
    out = lhs + rhs;
    return true;
  }

  if (is_sub2t(expr))
  {
    const auto &s = to_sub2t(expr);
    AffineExpr lhs, rhs;
    if (!try_as_affine(s.side_1, lhs) || !try_as_affine(s.side_2, rhs))
      return false;
    out = lhs - rhs;
    return true;
  }

  if (is_mul2t(expr))
  {
    // Linear only when at least one operand is a known constant.
    const auto &m = to_mul2t(expr);
    AffineExpr lhs, rhs;
    if (!try_as_affine(m.side_1, lhs) || !try_as_affine(m.side_2, rhs))
      return false;
    if (lhs.is_constant())
    {
      out = rhs * lhs.constant;
      return true;
    }
    if (rhs.is_constant())
    {
      out = lhs * rhs.constant;
      return true;
    }
    // non-linear: both operands contain variables
    return false;
  }

  return false;
}

expr2tc linear_equality_domaint::make_equality_expr(
  const irep_idt &lhs,
  const AffineExpr &rhs)
{
  const type2tc int_type = get_int_type(config.ansi_c.word_size);
  expr2tc rhs_expr = constant_int2tc(int_type, rhs.constant);

  for (const auto &kv : rhs.coeffs)
  {
    expr2tc var_expr = symbol2tc(int_type, kv.first);
    expr2tc term;
    if (kv.second == 1)
      term = var_expr;
    else if (kv.second == -1)
      term = neg2tc(int_type, var_expr);
    else
      term = mul2tc(int_type, constant_int2tc(int_type, kv.second), var_expr);

    // Skip adding 0 as the initial accumulator to keep the expression clean.
    if (is_constant_int2t(rhs_expr) && to_constant_int2t(rhs_expr).value == 0)
      rhs_expr = term;
    else
      rhs_expr = add2tc(int_type, rhs_expr, term);
  }

  return equality2tc(symbol2tc(int_type, lhs), rhs_expr);
}

const AffineExpr *
linear_equality_domaint::lookup(const irep_idt &var_name) const
{
  if (!equations)
    return nullptr;
  auto it = equations->find(var_name);
  return it == equations->end() ? nullptr : &it->second;
}

void linear_equality_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett to,
  ai_baset &,
  const namespacet &)
{
  if (is_bottom())
    return;

  const goto_programt::instructiont &insn = *from;

  switch (insn.type)
  {
  case ASSIGN:
  {
    if (!is_code_assign2t(insn.code))
      break;
    const code_assign2t &ca = to_code_assign2t(insn.code);
    if (!is_symbol2t(ca.target))
      // non-scalar lhs — leave state unchanged (conservative)
      break;

    const irep_idt &lhs_name = to_symbol2t(ca.target).thename;
    if (
      !is_signedbv_type(ca.target->type) &&
      !is_unsignedbv_type(ca.target->type) && !is_bool_type(ca.target->type))
    {
      // non-integer type — kill any equation for lhs
      forget(lhs_name);
      break;
    }

    AffineExpr rhs_affine;
    if (try_as_affine(ca.source, rhs_affine))
      assign_affine(lhs_name, rhs_affine);
    else
      // non-linear rhs — lose information about lhs
      forget(lhs_name);
    break;
  }

  case DECL:
    // Variable enters scope uninitialised — value is non-deterministic.
    if (is_code_decl2t(insn.code))
      forget(to_code_decl2t(insn.code).value);
    break;

  case DEAD:
    if (is_code_dead2t(insn.code))
      forget(to_code_dead2t(insn.code).value);
    break;

  case ASSUME:
  {
    // If the assumed condition is an equality x == expr, record it.
    if (!is_equality2t(insn.guard))
      break;
    const equality2t &eq = to_equality2t(insn.guard);
    AffineExpr affine;
    if (
      is_symbol2t(eq.side_1) && is_signedbv_type(eq.side_1->type) &&
      try_as_affine(eq.side_2, affine))
      assign_affine(to_symbol2t(eq.side_1).thename, affine);
    else if (
      is_symbol2t(eq.side_2) && is_signedbv_type(eq.side_2->type) &&
      try_as_affine(eq.side_1, affine))
      assign_affine(to_symbol2t(eq.side_2).thename, affine);
    break;
  }

  case FUNCTION_CALL:
    // The callee may write any value into the return variable.
    if (is_code_function_call2t(insn.code))
    {
      const code_function_call2t &call = to_code_function_call2t(insn.code);
      if (!is_nil_expr(call.ret) && is_symbol2t(call.ret))
        forget(to_symbol2t(call.ret).thename);
    }
    break;

  default:
    break;
  }

  (void)to;
}

bool linear_equality_domaint::merge(
  const linear_equality_domaint &src,
  goto_programt::const_targett,
  goto_programt::const_targett)
{
  if (src.is_bottom())
    // unreachable predecessor — nothing to add
    return false;

  if (is_bottom())
  {
    // First reachable predecessor — inherit its state.
    equations = src.equations;
    bottom = false;
    return true;
  }

  // Keep only equations both predecessors agree on (intersection / meet).
  bool changed = false;
  copy_if_needed();
  for (auto it = equations->begin(); it != equations->end();)
  {
    auto src_it = src.equations->find(it->first);
    if (src_it == src.equations->end() || src_it->second != it->second)
    {
      it = equations->erase(it);
      changed = true;
    }
    else
      ++it;
  }
  return changed;
}

void linear_equality_domaint::output(std::ostream &out) const
{
  if (is_bottom())
  {
    out << "  [BOTTOM]\n";
    return;
  }
  if (is_top())
  {
    out << "  [TOP]\n";
    return;
  }
  for (const auto &kv : *equations)
    out << "  " << kv.first << " == " << kv.second.to_string() << "\n";
}

expr2tc linear_equality_domaint::to_predicate() const
{
  if (is_bottom())
    return gen_false_expr();
  if (is_top())
    return gen_true_expr();

  std::vector<expr2tc> conjuncts;
  for (const auto &kv : *equations)
  {
    expr2tc eq = make_equality_expr(kv.first, kv.second);
    if (!is_nil_expr(eq))
      conjuncts.push_back(eq);
  }
  return conjuncts.empty() ? gen_true_expr() : conjunction(conjuncts);
}

expr2tc linear_equality_domaint::to_predicate(
  const std::unordered_set<irep_idt, dstring_hash> &vars) const
{
  if (is_bottom())
    return gen_false_expr();
  if (is_top())
    return gen_true_expr();

  std::vector<expr2tc> conjuncts;
  for (const auto &kv : *equations)
  {
    if (!vars.count(kv.first))
      // skip variables unrelated to the target loop
      continue;
    expr2tc eq = make_equality_expr(kv.first, kv.second);
    if (!is_nil_expr(eq))
      conjuncts.push_back(eq);
  }
  return conjuncts.empty() ? gen_true_expr() : conjunction(conjuncts);
}

bool linear_equality_domaint::ai_simplify(
  expr2tc &condition,
  const namespacet &) const
{
  if (is_bottom() || is_top() || !is_equality2t(condition))
    return true;

  const equality2t &eq = to_equality2t(condition);
  AffineExpr lhs_affine, rhs_affine;
  if (
    !try_as_affine(eq.side_1, lhs_affine) ||
    !try_as_affine(eq.side_2, rhs_affine))
    return true;

  AffineExpr diff = lhs_affine - rhs_affine;
  if (!diff.is_constant())
    // free variables remain — cannot decide
    return true;

  condition = (diff.constant == 0) ? gen_true_expr() : gen_false_expr();
  // condition was changed
  return false;
}
