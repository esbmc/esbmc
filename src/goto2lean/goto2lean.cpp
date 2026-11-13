#include <goto2lean/goto2lean.h>
#include <irep2/irep2_utils.h>
#include <util/arith/mp_arith.h>

#include <algorithm>
#include <cassert>
#include <regex>
#include <sstream>

namespace
{
const std::set<std::string> lean_keywords = {
  "at",
  "by",
  "calc",
  "def",
  "do",
  "else",
  "end",
  "example",
  "for",
  "from",
  "fun",
  "have",
  "if",
  "import",
  "in",
  "instance",
  "let",
  "match",
  "mut",
  "namespace",
  "open",
  "return",
  "section",
  "show",
  "then",
  "theorem",
  "unless",
  "variable",
  "where",
  "with",
  "Type",
  "Prop",
  "Sort",
  "structure",
  "inductive",
  "class",
  "true",
  "false",
  "True",
  "False",
  "abbrev",
  "axiom",
  "deriving",
  "using",
  "sorry",
  "forall",
  "exists",
  "extends",
  "mutual",
  "universe",
  "private",
  "partial",
  "unsafe",
  "local",
  "suffices",
  "nomatch",
  "try",
  "catch",
  "finally",
  "termination_by",
  "omit",
  "include",
  "opaque",
  "protected",
  "noncomputable",
  "decreasing_by",
  "fun_induction",
  "_"};

// Names the generated definitions use: continuations, join points and loops.
bool is_reserved(const std::string &name)
{
  static const std::regex generated("join_[0-9]+|.+_loop[0-9]+|Prog");
  return name == "post" || name == "ret" || name == "result" ||
         std::regex_match(name, generated);
}

std::string escape(const std::string &name)
{
  static const std::regex plain("[A-Za-z_][A-Za-z0-9_']*");
  return std::regex_match(name, plain) && !lean_keywords.count(name)
           ? name
           : "«" + name + "»";
}

// A Lean identifier for `base` not yet in `pool`: escaped with «» when it is
// not a plain identifier or is a keyword, and suffixed when taken or reserved.
std::string unique_ident(const std::string &base, std::set<std::string> &pool)
{
  std::string name = base;
  for (unsigned n = 1;; ++n)
  {
    std::string candidate = escape(name);
    if (!is_reserved(name) && !pool.count(candidate))
    {
      pool.insert(candidate);
      return candidate;
    }
    name = base + "_" + std::to_string(n);
  }
}

std::string shorthand(const std::string &id)
{
  std::string::size_type pos = id.rfind('@');
  return pos == std::string::npos ? id : id.substr(pos + 1);
}

std::string negate(const std::string &p)
{
  return p.rfind("¬", 0) == 0 ? p.substr(std::string("¬").size()) : "¬" + p;
}

std::string pad(unsigned ind)
{
  return std::string(2 * ind, ' ');
}

bool is_bitvector(const type2tc &t)
{
  return is_unsignedbv_type(t) || is_signedbv_type(t);
}

const expr2tc &operand(const expr2tc &e, unsigned k)
{
  return *e->get_sub_expr(k);
}

bool is_symbol_named(const expr2tc &e, const irep_idt &id)
{
  return is_symbol2t(e) && to_symbol2t(e).thename == id;
}

// ESBMC wraps C conditions in casts through an integer type, e.g.
// `(_Bool)(long)(!c)`; reading through them keeps the Lean goal propositional.
expr2tc strip_bool_roundtrip(const expr2tc &e)
{
  if (is_typecast2t(e) && is_bool_type(e->type))
  {
    const expr2tc &from = to_typecast2t(e).from;
    if (is_typecast2t(from) && is_bool_type(to_typecast2t(from).from->type))
      return strip_bool_roundtrip(to_typecast2t(from).from);
  }
  return e;
}

bool has_quantifier(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_forall2t(e) || is_exists2t(e))
    return true;
  bool found = false;
  e->foreach_operand(
    [&found](const expr2tc &op) { found = found || has_quantifier(op); });
  return found;
}

// The generated script introduces a ∀ that is the goal or a conjunct of it.
// An ∃ needs a witness and a quantified hypothesis an instantiation.
bool script_introduces_quantifiers(const expr2tc &raw)
{
  expr2tc e = strip_bool_roundtrip(raw);
  if (is_and2t(e))
    return script_introduces_quantifiers(to_and2t(e).side_1) &&
           script_introduces_quantifiers(to_and2t(e).side_2);
  if (is_forall2t(e))
    return script_introduces_quantifiers(to_forall2t(e).side_2);
  return !has_quantifier(e);
}

bool mentions(const expr2tc &e, const irep_idt &id)
{
  if (is_nil_expr(e))
    return false;
  if (is_symbol_named(e, id))
    return true;
  bool found = false;
  e->foreach_operand(
    [&](const expr2tc &op) { found = found || mentions(op, id); });
  return found;
}

// A literal, possibly negated (`-2` reaches the GOTO program as neg(2)).
bool constant_value(const expr2tc &e, BigInt &out)
{
  if (is_constant_int2t(e))
  {
    out = to_constant_int2t(e).value;
    return true;
  }
  if (is_neg2t(e) && constant_value(to_neg2t(e).value, out))
  {
    out = -out;
    return true;
  }
  return false;
}

expr2tc quantifier_binder(expr2tc e)
{
  while (is_typecast2t(e))
    e = to_typecast2t(e).from;
  if (is_address_of2t(e))
    e = to_address_of2t(e).ptr_obj;
  return e;
}

// `a < b`, `a <= b`, `a > b` or `a >= b`, read as `l < r` or `l ≤ r`.
struct comparisont
{
  const expr2tc &l, &r;
  bool strict;
};

comparisont ordered(const expr2tc &e)
{
  const expr2tc &a = operand(e, 0), &b = operand(e, 1);
  bool swap = is_greaterthan2t(e) || is_greaterthanequal2t(e);
  return {swap ? b : a, swap ? a : b, is_lessthan2t(e) || is_greaterthan2t(e)};
}

enum class effectt
{
  none,
  stop,
  write
};

// Whether `i` assigns `id`, may read it or leave straight-line code (stop),
// or neither.
effectt effect(const goto_programt::instructiont &i, const irep_idt &id)
{
  switch (i.type)
  {
  case DECL:
  case DEAD:
  case SKIP:
  case LOCATION:
    return effectt::none;
  case ASSERT:
  case ASSUME:
    return mentions(i.guard, id) ? effectt::stop : effectt::none;
  case ASSIGN:
  {
    const code_assign2t &a = to_code_assign2t(i.code);
    if (mentions(a.source, id))
      return effectt::stop;
    return is_symbol_named(a.target, id) ? effectt::write : effectt::none;
  }
  case FUNCTION_CALL:
  {
    const code_function_call2t &c = to_code_function_call2t(i.code);
    for (const expr2tc &arg : c.operands)
      if (mentions(arg, id))
        return effectt::stop;
    return is_symbol_named(c.ret, id) ? effectt::write : effectt::none;
  }
  default:
    return effectt::stop;
  }
}
} // namespace

goto2leant::goto2leant(
  const namespacet &ns,
  const goto_functionst &goto_functions,
  const std::string &entry,
  bool int_mode,
  bool div_checked)
  : ns(ns),
    goto_functions(goto_functions),
    entry(entry),
    int_mode(int_mode),
    div_checked(div_checked)
{
}

void goto2leant::unsupported(const std::string &what, const locationt &loc)
  const
{
  std::string where;
  if (!loc.get_file().empty())
    where =
      " at " + loc.get_file().as_string() + ":" + loc.get_line().as_string();
  throw goto2lean_unsupported("goto2lean: unsupported " + what + where);
}

std::string goto2leant::fresh(const std::string &base)
{
  return unique_ident(base, cur->used_names);
}

// C++ ids carry signatures (`c:@F@main#`); the symbol's name is the source one.
std::string goto2leant::source_name(const irep_idt &id) const
{
  const symbolt *s = ns.lookup(id);
  // Python nests helpers as `check@F@mid`; the last component is the name.
  return shorthand(s ? s->name.as_string() : id.as_string());
}

std::string goto2leant::var_name(const irep_idt &id)
{
  auto it = cur->names.find(id.as_string());
  if (it != cur->names.end())
    return it->second;
  const symbolt *s = ns.lookup(id);
  if (s && s->static_lifetime)
    unsupported("global variable " + id.as_string(), s->location);
  std::string base = s ? s->name.as_string() : shorthand(id.as_string());
  if (auto cut = base.find("::"); cut != std::string::npos)
    base.resize(cut); // C++ parameters: `lo::0`
  return cur->names[id.as_string()] = fresh(base);
}

std::string goto2leant::type(const type2tc &t)
{
  if (is_bool_type(t))
    return "Bool";
  if (is_bitvector(t))
    return int_mode ? "Int" : "BitVec " + std::to_string(t->get_width());
  if (is_empty_type(t))
    return "Unit";
  unsupported("type " + get_type_id(t));
}

std::string goto2leant::ret_cont(const loopt *loop) const
{
  return loop ? "ret" : "post";
}

std::string goto2leant::args_of(const scopet &scope)
{
  std::string out;
  for (const vart &v : scope)
    out += " " + v.name;
  return out;
}

std::string goto2leant::lambda(const scopet &scope, const std::string &body)
{
  if (scope.empty())
    return "(\n" + body + ")";
  std::string out = "(fun";
  for (const vart &v : scope)
    out += " (" + v.name + " : " + type(v.type) + ")";
  return out + " =>\n" + body + ")";
}

std::string goto2leant::zero(const type2tc &t) const
{
  return int_mode ? "(0 : Int)" : "0#" + std::to_string(t->get_width());
}

std::string
goto2leant::infix(const expr2tc &a, const std::string &op, const expr2tc &b)
{
  return "(" + value(a) + " " + op + " " + value(b) + ")";
}

std::string goto2leant::infix(const expr2tc &e, const std::string &op)
{
  return infix(operand(e, 0), op, operand(e, 1));
}

std::string
goto2leant::apply(const std::string &fn, const expr2tc &a, const expr2tc &b)
{
  return "(" + fn + " " + value(a) + " " + value(b) + ")";
}

void goto2leant::reject_under_ir(const expr2tc &e) const
{
  if (int_mode)
    unsupported(get_expr_id(e) + " under --ir (unbounded integers)");
}

std::string goto2leant::value(const expr2tc &e)
{
  switch (e->expr_id)
  {
  case expr2t::symbol_id:
    return symbol_value(to_symbol2t(e).thename);
  case expr2t::constant_bool_id:
    return to_constant_bool2t(e).value ? "true" : "false";
  case expr2t::constant_int_id:
    return int_literal(e);
  case expr2t::add_id:
    return infix(e, "+");
  case expr2t::sub_id:
    return infix(e, "-");
  case expr2t::mul_id:
    return infix(e, "*");
  case expr2t::div_id:
  case expr2t::modulus_id:
    return int_mode ? int_division(e) : bv_division(e);
  case expr2t::neg_id:
    return "(-" + value(to_neg2t(e).value) + ")";
  case expr2t::bitnot_id:
  case expr2t::bitand_id:
  case expr2t::bitor_id:
  case expr2t::bitxor_id:
  case expr2t::shl_id:
  case expr2t::lshr_id:
  case expr2t::ashr_id:
    return bitwise(e);
  case expr2t::if_id:
  {
    const if2t &i = to_if2t(e);
    return "(if " + value(i.cond) + " then " + value(i.true_value) + " else " +
           value(i.false_value) + ")";
  }
  case expr2t::typecast_id:
    return cast(e);
  case expr2t::equality_id:
    return infix(e, "==");
  case expr2t::notequal_id:
  case expr2t::xor_id:
    return infix(e, "!=");
  case expr2t::lessthan_id:
  case expr2t::lessthanequal_id:
  case expr2t::greaterthan_id:
  case expr2t::greaterthanequal_id:
    return relation(e);
  case expr2t::not_id:
    return "(!" + value(to_not2t(e).value) + ")";
  case expr2t::and_id:
    return infix(e, "&&");
  case expr2t::or_id:
    return infix(e, "||");
  case expr2t::overflow_id:
    return overflow(e);
  case expr2t::overflow_neg_id:
    return overflow_neg(e);
  case expr2t::forall_id:
  case expr2t::exists_id:
    unsupported(
      "quantifier used as a value (only guards of ASSERT, ASSUME and GOTO)");
  default:
    unsupported("expression " + get_expr_id(e));
  }
}

std::string goto2leant::symbol_value(const irep_idt &id)
{
  // A read the current path has not bound (e.g. after a jump past the
  // declaration) would silently capture an outer Lean binding.
  bool bound =
    std::any_of(
      in_scope->begin(),
      in_scope->end(),
      [&](const vart &v) { return v.id == id; }) ||
    std::find(qbinders.begin(), qbinders.end(), id) != qbinders.end();
  if (!bound)
    unsupported(
      "read of " + shorthand(id.as_string()) + " where it is not bound");
  return var_name(id);
}

std::string goto2leant::int_literal(const expr2tc &e) const
{
  const BigInt &v = to_constant_int2t(e).value;
  if (int_mode)
    return "(" + integer2string(v) + " : Int)";
  const std::string w = std::to_string(e->type->get_width());
  if (v.is_negative())
    return "(BitVec.ofInt " + w + " (" + integer2string(v) + "))";
  return "(" + integer2string(v) + "#" + w + ")";
}

std::string goto2leant::int_division(const expr2tc &e)
{
  // --ir truncates signed / and % (mk_truncated_div/rem) but keeps unsigned
  // ones Euclidean (smt_solver.cpp): Int.tdiv/tmod and Lean's / and %. SMT
  // leaves a zero divisor unspecified, so it must be ruled out by ESBMC.
  const expr2tc &a = operand(e, 0), &b = operand(e, 1);
  BigInt c;
  bool nonzero_constant = constant_value(b, c) && !c.is_zero();
  if (!div_checked && !nonzero_constant)
    unsupported(
      "division by a variable under --ir with --no-div-by-zero-check");
  if (!is_signedbv_type(e->type))
    return infix(a, is_div2t(e) ? "/" : "%", b);
  if (nonzero_constant)
    divisor_signs[integer2string(c)] = c.is_negative() ? -1 : 1;
  return apply(is_div2t(e) ? "Int.tdiv" : "Int.tmod", a, b);
}

std::string goto2leant::bv_division(const expr2tc &e)
{
  const expr2tc &a = operand(e, 0), &b = operand(e, 1);
  bool is_signed = is_signedbv_type(e->type);
  // ESBMC lowers `/` to SMT-LIB bvudiv/bvsdiv (smt_solver.cpp); Lean's
  // smtUDiv/smtSDiv agree with them, including on a zero divisor.
  if (is_div2t(e))
    return apply(is_signed ? "BitVec.smtSDiv" : "BitVec.smtUDiv", a, b);
  return is_signed ? apply("BitVec.srem", a, b) : infix(a, "%", b);
}

std::string goto2leant::bitwise(const expr2tc &e)
{
  reject_under_ir(e);
  switch (e->expr_id)
  {
  case expr2t::bitnot_id:
    return "(~~~" + value(to_bitnot2t(e).value) + ")";
  case expr2t::bitand_id:
    return infix(e, "&&&");
  case expr2t::bitor_id:
    return infix(e, "|||");
  case expr2t::bitxor_id:
    return infix(e, "^^^");
  case expr2t::shl_id:
    return infix(e, "<<<");
  case expr2t::lshr_id:
    return infix(e, ">>>");
  default:
    assert(is_ashr2t(e));
    return apply("BitVec.sshiftRight'", operand(e, 0), operand(e, 1));
  }
}

std::string goto2leant::cast(const expr2tc &e)
{
  const type2tc &t = e->type;
  const expr2tc &from = to_typecast2t(e).from;
  if (is_bool_type(t) && is_bitvector(from->type))
    return "(" + value(from) + " != " + zero(from->type) + ")";
  if (is_bitvector(t) && is_bool_type(from->type))
  {
    const std::string one =
      int_mode ? "(1 : Int)" : "1#" + std::to_string(t->get_width());
    return "(if " + value(from) + " then " + one + " else " + zero(t) + ")";
  }
  if (!is_bitvector(t) || !is_bitvector(from->type))
    unsupported(
      "cast from " + get_type_id(from->type) + " to " + get_type_id(t));
  unsigned to_w = t->get_width(), from_w = from->type->get_width();
  // Integer casts are the identity under --ir
  // (convert_typecast_to_ints_intmode).
  if (to_w == from_w || int_mode)
    return value(from);
  const char *fn = to_w > from_w && is_signedbv_type(from->type)
                     ? "BitVec.signExtend"
                     : "BitVec.setWidth";
  return "(" + std::string(fn) + " " + std::to_string(to_w) + " " +
         value(from) + ")";
}

std::string goto2leant::relation(const expr2tc &e)
{
  const comparisont c = ordered(e);
  if (int_mode)
    return "(decide (" + value(c.l) + (c.strict ? " < " : " ≤ ") + value(c.r) +
           "))";
  const char *fn = is_signedbv_type(c.l->type)
                     ? (c.strict ? "BitVec.slt" : "BitVec.sle")
                     : (c.strict ? "BitVec.ult" : "BitVec.ule");
  return apply(fn, c.l, c.r);
}

std::string goto2leant::overflow(const expr2tc &e)
{
  reject_under_ir(e);
  const expr2tc &op = to_overflow2t(e).operand;
  const char *u, *s;
  switch (op->expr_id)
  {
  case expr2t::add_id:
    u = "BitVec.uaddOverflow", s = "BitVec.saddOverflow";
    break;
  case expr2t::sub_id:
    u = "BitVec.usubOverflow", s = "BitVec.ssubOverflow";
    break;
  case expr2t::mul_id:
    u = "BitVec.umulOverflow", s = "BitVec.smulOverflow";
    break;
  case expr2t::div_id:
  case expr2t::modulus_id:
  {
    // As smt_overflow.cpp: signed INT_MIN / -1, unsigned a zero divisor.
    const expr2tc &b = operand(op, 1);
    if (is_signedbv_type(operand(op, 0)->type))
      return apply("BitVec.sdivOverflow", operand(op, 0), b);
    return "(" + value(b) + " == " + zero(b->type) + ")";
  }
  default:
    unsupported("overflow check on " + get_expr_id(op));
  }
  const expr2tc &a = operand(op, 0);
  return apply(is_signedbv_type(a->type) ? s : u, a, operand(op, 1));
}

std::string goto2leant::overflow_neg(const expr2tc &e)
{
  reject_under_ir(e);
  const expr2tc &op = to_overflow_neg2t(e).operand;
  const std::string w = std::to_string(op->type->get_width());
  return is_signedbv_type(op->type)
           ? "(" + value(op) + " == BitVec.intMin " + w + ")"
           : "(" + value(op) + " != " + zero(op->type) + ")";
}

std::string goto2leant::prop(const expr2tc &raw)
{
  expr2tc e = strip_bool_roundtrip(raw);
  switch (e->expr_id)
  {
  case expr2t::constant_bool_id:
    return to_constant_bool2t(e).value ? "True" : "False";
  case expr2t::not_id:
  {
    expr2tc inner = strip_bool_roundtrip(to_not2t(e).value);
    if (is_not2t(inner))
      return prop(to_not2t(inner).value);
    return "¬" + prop(inner);
  }
  case expr2t::and_id:
    return connective(e, "∧");
  case expr2t::or_id:
    return connective(e, "∨");
  case expr2t::implies_id:
    return connective(e, "→");
  case expr2t::equality_id:
    return infix(e, "=");
  case expr2t::notequal_id:
    return infix(e, "≠");
  case expr2t::lessthan_id:
  case expr2t::lessthanequal_id:
  case expr2t::greaterthan_id:
  case expr2t::greaterthanequal_id:
    return prop_relation(e);
  case expr2t::forall_id:
  case expr2t::exists_id:
    return quantifier(e);
  case expr2t::typecast_id:
  {
    const expr2tc &from = to_typecast2t(e).from;
    if (is_bitvector(from->type))
      return "(" + value(from) + " ≠ " + zero(from->type) + ")";
    break;
  }
  default:
    break;
  }
  assert(is_bool_type(e->type));
  return "(" + value(e) + " = true)";
}

std::string goto2leant::connective(const expr2tc &e, const std::string &op)
{
  return "(" + prop(operand(e, 0)) + " " + op + " " + prop(operand(e, 1)) + ")";
}

std::string goto2leant::prop_relation(const expr2tc &e)
{
  const comparisont c = ordered(e);
  if (is_signedbv_type(operand(e, 0)->type) && !int_mode)
    return "(" + std::string(c.strict ? "BitVec.slt" : "BitVec.sle") + " " +
           value(c.l) + " " + value(c.r) + " = true)";
  return infix(c.l, c.strict ? "<" : "≤", c.r);
}

std::string goto2leant::quantifier(const expr2tc &e)
{
  manual_quantifiers = true;
  bool all = is_forall2t(e);
  const expr2tc &binder =
    quantifier_binder(all ? to_forall2t(e).side_1 : to_exists2t(e).side_1);
  const expr2tc &body = all ? to_forall2t(e).side_2 : to_exists2t(e).side_2;
  if (!is_symbol2t(binder))
    unsupported("quantifier binder " + get_expr_id(binder));
  qbinders.push_back(to_symbol2t(binder).thename);
  std::string inner = prop(body);
  qbinders.pop_back();
  return "(" + std::string(all ? "∀ " : "∃ ") +
         var_name(to_symbol2t(binder).thename) + " : " + type(binder->type) +
         ", " + inner + ")";
}

std::string goto2leant::jump(
  std::size_t target,
  std::size_t end,
  const scopet &scope,
  const loopt *loop,
  const std::string &on_end,
  unsigned ind)
{
  if (loop && target == loop->head)
    return pad(ind) + loop->fn + args_of(loop->vars) + " post ret";
  if (loop && target == loop->exit)
    return pad(ind) + "post" + args_of(loop->vars);
  if (target == end)
  {
    assert(!on_end.empty()); // only the function body's region has none
    return pad(ind) + on_end;
  }
  if (target < end)
    return trans(target, end, scope, loop, on_end, ind);
  // target > end, and every target is an instruction, so end < insts.size().
  unsupported("unstructured jump out of a branch", cur->insts[end]->location);
}

std::string goto2leant::trans_loop(
  std::size_t pc,
  std::size_t end,
  const scopet &scope,
  const loopt *outer,
  const std::string &on_end,
  unsigned ind)
{
  // Lean cannot show a fixpoint monotone through another fixpoint's
  // continuation, so nested loops would not typecheck.
  if (outer)
    unsupported("nested loop", cur->insts[pc]->location);
  // Unique: function names are, and user names never match `*_loopN`.
  std::string fn =
    escape(cur->source + "_loop" + std::to_string(++cur->loop_count));
  global_names.insert(fn);
  loopt loop{pc, cur->loop_last.at(pc) + 1, fn, live(scope, pc, outer)};
  if (loop.exit > end)
    unsupported(
      "loop that leaves its enclosing branch", cur->insts[pc]->location);
  has_loops = true;

  std::string post_type, params;
  for (const vart &v : loop.vars)
  {
    params += " (" + v.name + " : " + type(v.type) + ")";
    post_type += type(v.type) + " → ";
  }
  std::ostringstream def;
  def << "-- loop at " << cur->insts[pc]->location.get_file() << ":"
      << cur->insts[pc]->location.get_line() << "\n"
      << "def " << loop.fn << params << "\n    (post : " << post_type
      << "Prop) (ret : " << type(cur->ret_type) << " → Prop) : Prop :=\n"
      << trans(pc, loop.exit, loop.vars, &loop, "post" + args_of(loop.vars), 1)
      << "\n"
      << "coinductive_fixpoint\n";
  aux_defs.push_back(def.str());

  std::string rest = trans(loop.exit, end, loop.vars, outer, on_end, ind + 1);
  return pad(ind) + loop.fn + args_of(loop.vars) + " " +
         lambda(loop.vars, rest) + " " + ret_cont(outer);
}

// Expressions are translated against this path's bindings, before recursing.
std::string goto2leant::value_in(const scopet &scope, const expr2tc &e)
{
  in_scope = &scope;
  return value(e);
}

std::string goto2leant::prop_in(const scopet &scope, const expr2tc &e)
{
  in_scope = &scope;
  return prop(e);
}

std::string goto2leant::asserted(const scopet &scope, const expr2tc &guard)
{
  const bool manual = manual_quantifiers;
  std::string g = prop_in(scope, guard);
  if (script_introduces_quantifiers(guard))
    manual_quantifiers = manual;
  return g;
}

std::string goto2leant::bind(scopet &scope, const expr2tc &sym)
{
  const irep_idt &id = to_symbol2t(sym).thename;
  for (const vart &v : scope)
    if (v.id == id)
      return v.name;
  scope.push_back(vart{var_name(id), sym->type, id});
  return scope.back().name;
}

std::string goto2leant::trans(
  std::size_t pc,
  std::size_t end,
  scopet scope,
  const loopt *loop,
  const std::string &on_end,
  unsigned ind)
{
  const std::string p = pad(ind);
  if (pc == end)
  {
    assert(!on_end.empty()); // RETURN and END_FUNCTION end the function body
    return p + on_end;
  }
  assert(pc < end);
  if (!(loop && loop->head == pc) && cur->loop_last.count(pc))
    return trans_loop(pc, end, scope, loop, on_end, ind);

  const instructiont &I = *cur->insts[pc];
  auto next = [&]() { return trans(pc + 1, end, scope, loop, on_end, ind); };

  switch (I.type)
  {
  case SKIP:
  case LOCATION:
  case DEAD:
    return next();

  case OTHER:
    // An expression statement is evaluated for its checks only; value()
    // rejects what symex would check (dereferences, arrays). Anything else
    // (printf, free, ...) carries semantics this translation does not model.
    if (!is_code_expression2t(I.code))
      unsupported("statement " + get_expr_id(I.code), I.location);
    value_in(scope, to_code_expression2t(I.code).operand);
    return next();

  case DECL:
    return trans_decl(pc, end, scope, loop, on_end, ind);

  case ASSIGN:
    return trans_assign(pc, end, scope, loop, on_end, ind);

  case ASSUME:
  {
    std::string g = prop_in(scope, I.guard);
    return p + g + " →\n" + next();
  }

  case ASSERT:
  {
    std::string g = asserted(scope, I.guard);
    return p + "-- " + I.location.comment().as_string() + "\n" + p + g +
           " ∧ (\n" + next() + ")";
  }

  case FUNCTION_CALL:
    return trans_call(pc, end, scope, loop, on_end, ind);

  case RETURN:
  {
    const code_return2t &r = to_code_return2t(I.code);
    return p + ret_cont(loop) +
           (is_nil_expr(r.operand) ? " ()" : " " + value_in(scope, r.operand));
  }

  case END_FUNCTION:
    if (is_empty_type(cur->ret_type))
      return p + ret_cont(loop) + " ()";
    return p + "∀ result : " + type(cur->ret_type) + ", " + ret_cont(loop) +
           " result";

  case GOTO:
    return trans_goto(pc, end, scope, loop, on_end, ind);

  default:
    unsupported(
      "instruction type " + std::to_string(static_cast<int>(I.type)),
      I.location);
  }
}

std::string goto2leant::trans_decl(
  std::size_t pc,
  std::size_t end,
  scopet &scope,
  const loopt *loop,
  const std::string &on_end,
  unsigned ind)
{
  const code_decl2t &d = to_code_decl2t(cur->insts[pc]->code);
  vart v{var_name(d.value), d.type, d.value};
  std::string t = type(v.type);
  scope.erase(
    std::remove_if(
      scope.begin(), scope.end(), [&](const vart &w) { return w.id == v.id; }),
    scope.end());
  scope.push_back(v);
  // goto_convert emits declarations without initialisers (the value
  // follows as an assignment), so an unassigned declaration is nondet.
  assert(is_nil_expr(d.init));
  std::string head;
  if (!assigned_before_use(pc + 1, end, d.value))
    head = pad(ind) + "∀ " + v.name + " : " + t + ",\n";
  return head + trans(pc + 1, end, scope, loop, on_end, ind);
}

std::string goto2leant::trans_assign(
  std::size_t pc,
  std::size_t end,
  scopet &scope,
  const loopt *loop,
  const std::string &on_end,
  unsigned ind)
{
  const std::string p = pad(ind);
  const instructiont &I = *cur->insts[pc];
  const code_assign2t &a = to_code_assign2t(I.code);
  if (!is_symbol2t(a.target))
    unsupported("assignment to " + get_expr_id(a.target), I.location);
  bool nondet = is_sideeffect2t(a.source) &&
                to_sideeffect2t(a.source).kind == sideeffect_allockind::nondet;
  if (!nondet)
  {
    std::string rhs = value_in(scope, a.source);
    std::string x = bind(scope, a.target);
    return p + "let " + x + " := " + rhs + ";\n" +
           trans(pc + 1, end, scope, loop, on_end, ind);
  }
  std::string x = bind(scope, a.target);

  // `tmp = NONDET; v = tmp;` binds `v` directly, so Lean's counterexamples
  // name the program variable rather than ESBMC's temporary.
  if (pc + 1 < end && cur->insts[pc + 1]->type == ASSIGN)
  {
    const code_assign2t &b = to_code_assign2t(cur->insts[pc + 1]->code);
    if (
      is_symbol2t(b.target) && b.source == a.target &&
      b.target->type == a.target->type)
    {
      std::string v = bind(scope, b.target);
      return p + "∀ " + v + " : " + type(b.target->type) + ",\n" + p + "let " +
             x + " := " + v + ";\n" +
             trans(pc + 2, end, scope, loop, on_end, ind);
    }
  }
  return p + "∀ " + x + " : " + type(a.target->type) + ",\n" +
         trans(pc + 1, end, scope, loop, on_end, ind);
}

std::string goto2leant::trans_call(
  std::size_t pc,
  std::size_t end,
  scopet &scope,
  const loopt *loop,
  const std::string &on_end,
  unsigned ind)
{
  const std::string p = pad(ind);
  const instructiont &I = *cur->insts[pc];
  const code_function_call2t &c = to_code_function_call2t(I.code);
  if (!is_symbol2t(c.function))
    unsupported("indirect function call", I.location);
  const std::string callee = to_symbol2t(c.function).thename.as_string();
  if (!is_nil_expr(c.ret) && !is_symbol2t(c.ret))
    unsupported("call result stored in " + get_expr_id(c.ret), I.location);
  std::string args;
  for (const expr2tc &arg : c.operands)
    args += " " + value_in(scope, arg); // also rejects what symex would check
  auto it = fn_names.find(callee);
  if (it == fn_names.end())
  {
    // ESBMC intrinsics have semantics of their own (e.g. unreachability).
    const std::string name = shorthand(callee);
    if (name.rfind("__ESBMC", 0) == 0 || name.rfind("__VERIFIER", 0) == 0)
      unsupported("call to intrinsic " + name, I.location);
    // No body: the only effect modelled is a nondet result.
    if (is_nil_expr(c.ret))
      return p + "-- call to " + name + " (no body): ignored\n" +
             trans(pc + 1, end, scope, loop, on_end, ind);
    std::string r = bind(scope, c.ret);
    return p + "-- call to " + name + " (no body): nondet result\n" + p + "∀ " +
           r + " : " + type(c.ret->type) + ",\n" +
           trans(pc + 1, end, scope, loop, on_end, ind);
  }
  // As with nested loops, a fixpoint is not monotone through a callee.
  if (loop)
    unsupported("call to " + shorthand(callee) + " inside a loop", I.location);
  std::string binder = is_nil_expr(c.ret) ? "_" : bind(scope, c.ret);
  return p + it->second + args + " (fun " + binder + " =>\n" +
         trans(pc + 1, end, scope, loop, on_end, ind) + ")";
}

// if (!g) { then } [else { els }]: the else branch, when present, is the
// range [t, join) skipped by the unconditional jump that ends `then`.
std::size_t
goto2leant::branch_join(std::size_t pc, std::size_t t, std::size_t end) const
{
  const instructiont &before = *cur->insts[t - 1];
  if (t - 1 > pc && before.is_goto() && is_true(before.guard))
  {
    std::size_t e = cur->index.at(&*before.targets.front());
    if (e > t && e <= end)
      return e;
  }
  return t;
}

std::string goto2leant::trans_goto(
  std::size_t pc,
  std::size_t end,
  const scopet &scope,
  const loopt *loop,
  const std::string &on_end,
  unsigned ind)
{
  const std::string p = pad(ind);
  const instructiont &I = *cur->insts[pc];
  std::size_t t = cur->index.at(&*I.targets.front());
  if (is_true(I.guard))
    return jump(t, end, scope, loop, on_end, ind);
  std::string g = prop_in(scope, I.guard);

  // A jump to the enclosing loop's head or exit, or out of this region.
  if ((loop && (t == loop->head || t == loop->exit)) || t >= end)
    return p + "(" + g + " →\n" + jump(t, end, scope, loop, on_end, ind + 1) +
           ") ∧\n" + p + "(" + negate(g) + " →\n" +
           trans(pc + 1, end, scope, loop, on_end, ind) + ")";
  if (t <= pc)
    unsupported("backward conditional jump", I.location);

  std::size_t join = branch_join(pc, t, end);
  std::size_t then_end = join == t ? t : t - 1;
  std::string head, call_join = on_end;
  if (join != end)
  {
    std::string name = "join_" + std::to_string(++join_count);
    scopet params = live(scope, join, loop);
    call_join = name + args_of(params);
    head = p + "let " + name + " := " +
           lambda(params, trans(join, end, params, loop, on_end, ind + 1)) +
           ";\n";
  }
  std::string taken = join == t
                        ? pad(ind + 1) + call_join
                        : trans(t, join, scope, loop, call_join, ind + 1);
  std::string fall = trans(pc + 1, then_end, scope, loop, call_join, ind + 1);
  return head + p + "(" + g + " →\n" + taken + ") ∧\n" + p + "(" + negate(g) +
         " →\n" + fall + ")";
}

// The variables of `scope` still needed from `pc` on: read or written later,
// or carried around the enclosing loop.
goto2leant::scopet
goto2leant::live(const scopet &scope, std::size_t pc, const loopt *loop) const
{
  scopet out;
  for (const vart &v : scope)
  {
    bool keep = false;
    if (loop)
      for (const vart &l : loop->vars)
        keep = keep || l.id == v.id;
    for (std::size_t k = pc; !keep && k < cur->insts.size(); ++k)
      keep = mentions(cur->insts[k]->code, v.id) ||
             mentions(cur->insts[k]->guard, v.id);
    if (keep)
      out.push_back(v);
  }
  return out;
}

// True when straight-line code from `pc` binds `id` before anything could
// read it, so the uninitialised declaration needs no `∀` binder of its own.
bool goto2leant::assigned_before_use(
  std::size_t pc,
  std::size_t end,
  const irep_idt &id) const
{
  for (;; ++pc)
  {
    if (
      pc == end || cur->insts[pc]->is_target() ||
      cur->insts[pc]->is_backwards_goto())
      return false;
    effectt first = effect(*cur->insts[pc], id);
    if (first != effectt::none)
      return first == effectt::write;
  }
}

void goto2leant::collect(const std::string &id, std::set<std::string> &on_stack)
{
  if (fn_names.count(id))
    return;
  auto it = goto_functions.function_map.find(id);
  if (it == goto_functions.function_map.end() || !it->second.body_available)
    return;
  if (!on_stack.insert(id).second)
    unsupported("recursive function " + shorthand(id));
  for (const auto &i : it->second.body.instructions)
    if (i.is_function_call())
    {
      const expr2tc &f = to_code_function_call2t(i.code).function;
      if (is_symbol2t(f))
        collect(to_symbol2t(f).thename.as_string(), on_stack);
    }
  on_stack.erase(id);
  fn_names[id] = unique_ident(source_name(id), global_names);
  order.push_back(id);
}

std::string goto2leant::translate_function(const std::string &id)
{
  const goto_functiont &gf = goto_functions.function_map.at(id);
  functiont f;
  f.lean_name = fn_names.at(id);
  f.source = source_name(id);
  f.used_names = global_names; // locals must not shadow functions or loops
  for (const auto &i : gf.body.instructions)
  {
    f.index[&i] = f.insts.size();
    f.insts.push_back(&i);
  }
  for (std::size_t j = 0; j < f.insts.size(); ++j)
    if (f.insts[j]->is_backwards_goto())
    {
      std::size_t head = f.index.at(&*f.insts[j]->targets.front());
      f.loop_last[head] = std::max(f.loop_last[head], j);
    }
  const code_type2t &ft = to_code_type(gf.type);
  f.ret_type = ft.ret_type;
  cur = &f;

  scopet params;
  std::string header = "def " + f.lean_name;
  for (std::size_t k = 0; k < ft.arguments.size(); ++k)
  {
    vart v{
      var_name(ft.argument_names[k]), ft.arguments[k], ft.argument_names[k]};
    header += " (" + v.name + " : " + type(v.type) + ")";
    params.push_back(v);
  }
  if (id == entry_id)
    entry_params = params;
  header += " (post : " + type(f.ret_type) + " → Prop) : Prop :=\n";

  aux_defs.clear();
  std::string body = trans(0, f.insts.size(), params, nullptr, "", 1);
  std::string out;
  for (const std::string &d : aux_defs)
    out += d + "\n";
  cur = nullptr;
  return out + header + body + "\n";
}

// ESBMC runs start-up code before the entry (C++ static constructors, Python
// module code); its assertions are not part of this translation.
void goto2leant::check_startup_code() const
{
  std::vector<std::string> todo = {"__ESBMC_main"};
  std::set<std::string> seen;
  while (!todo.empty())
  {
    std::string id = todo.back();
    todo.pop_back();
    auto it = goto_functions.function_map.find(id);
    // ESBMC's own runtime hooks (e.g. thread counters) are part of its
    // trusted model; __ESBMC_main itself can hold checks on the program.
    bool runtime =
      id != "__ESBMC_main" && source_name(id).rfind("__ESBMC_", 0) == 0;
    if (
      id == entry_id || runtime || !seen.insert(id).second ||
      it == goto_functions.function_map.end())
      continue;
    for (const auto &i : it->second.body.instructions)
      if (i.is_assert())
        unsupported(
          "assertion in start-up code " + source_name(id) +
            " (static constructor or module code)",
          i.location);
      else if (
        i.is_function_call() &&
        is_symbol2t(to_code_function_call2t(i.code).function))
        todo.push_back(to_symbol2t(to_code_function_call2t(i.code).function)
                         .thename.as_string());
  }
}

std::string goto2leant::proof_script() const
{
  std::ostringstream out;
  if (has_loops || manual_quantifiers)
  {
    out << "  -- Not closed automatically: "
        << (has_loops
              ? "each loop needs an invariant (apply Prog.<loop>.coinduct)"
              : "only an asserted ∀ is introduced automatically; any other "
                "quantifier needs a manual proof")
        << ".\n  sorry\n";
    return out.str();
  }
  out << "  simp only [";
  for (std::size_t k = 0; k < order.size(); ++k)
    out << (k ? ", " : "") << "Prog." << fn_names.at(order[k]);
  out << "]\n";
  // Unbounded integers: omega decides linear Int arithmetic once truncated
  // division is rewritten into the Euclidean / and % it understands.
  if (int_mode)
  {
    // omega cannot evaluate Int.sign, so state it for each constant divisor.
    out << "  try simp only [Int.tdiv_eq_ediv, Int.tmod_eq_emod, "
           "Int.dvd_iff_emod_eq_zero";
    for (const auto &[c, sign] : divisor_signs)
      out << ",\n    show Int.sign (" << c << " : Int) = " << sign
          << " from rfl";
    out << "]\n";
  }
  out << "  repeat' (first | apply And.intro | intro)\n"
      << (int_mode
            ? "  all_goals (repeat' split) <;> (try simp_all) <;> omega\n"
            : "  all_goals bv_decide\n");
  return out.str();
}

std::string goto2leant::translate()
{
  for (const auto &[id, fn] : goto_functions.function_map)
    if (fn.body_available && source_name(id) == entry && entry_id.empty())
      entry_id = id.as_string();
  if (entry_id.empty())
    throw goto2lean_unsupported(
      "goto2lean: entry function " + entry + " not found");

  check_startup_code();

  std::set<std::string> on_stack;
  collect(entry_id, on_stack);

  std::string defs;
  for (const std::string &id : order)
    defs += translate_function(id) + "\n";

  std::string binders, args;
  for (const vart &v : entry_params)
  {
    binders += " (" + v.name + " : " + type(v.type) + ")";
    args += " " + v.name;
  }
  const std::string entry_fn = "Prog." + fn_names.at(entry_id) + args;
  std::ostringstream out;
  out << "-- Generated by ESBMC --goto2lean (prototype).\n"
      << "-- Each function takes a postcondition `post`; NONDET is ∀, ASSUME "
         "is →,\n"
      << "-- ASSERT is ∧, and loops are greatest fixpoints (partial "
         "correctness).\n"
      << "import Std.Tactic.BVDecide\n"
      << "set_option linter.unusedVariables false\n\n"
      << "namespace Prog\n\n"
      << defs << "end Prog\n\n"
      << "/-- No ESBMC assertion fails on any terminating execution of "
      << fn_names.at(entry_id) << ". -/\n"
      << "theorem " << fn_names.at(entry_id)
      << "_safe :" << (binders.empty() ? "" : " ∀" + binders + ",") << " "
      << entry_fn << " (fun _ => True) := by\n"
      << proof_script();
  return out.str();
}
