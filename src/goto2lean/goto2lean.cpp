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

bool mentions(const expr2tc &e, const irep_idt &id)
{
  if (is_nil_expr(e))
    return false;
  if (is_symbol2t(e) && to_symbol2t(e).thename == id)
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
} // namespace

goto2leant::goto2leant(
  const namespacet &ns,
  const goto_functionst &goto_functions,
  const std::string &entry,
  bool int_mode,
  bool div_checked,
  bool simplifier_on)
  : ns(ns),
    goto_functions(goto_functions),
    entry(entry),
    int_mode(int_mode),
    div_checked(div_checked),
    simplifier_on(simplifier_on)
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
  base = base.substr(0, base.find("::")); // C++ parameters: `lo::0`
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

std::string goto2leant::value(const expr2tc &e)
{
  const type2tc &t = e->type;
  auto bin = [this](const expr2tc &a, const char *op, const expr2tc &b) {
    return "(" + value(a) + " " + op + " " + value(b) + ")";
  };
  auto call = [this](const char *fn, const expr2tc &a, const expr2tc &b) {
    return "(" + std::string(fn) + " " + value(a) + " " + value(b) + ")";
  };

  if (is_symbol2t(e))
  {
    // A read the current path has not bound (e.g. after a jump past the
    // declaration) would silently capture an outer Lean binding.
    const irep_idt &id = to_symbol2t(e).thename;
    bool bound = false;
    for (const vart &v : *in_scope)
      bound = bound || v.id == id;
    for (const irep_idt &q : qbinders)
      bound = bound || q == id;
    if (!bound)
      unsupported(
        "read of " + shorthand(id.as_string()) + " where it is not bound");
    return var_name(id);
  }
  if (is_constant_bool2t(e))
    return to_constant_bool2t(e).value ? "true" : "false";
  if (is_constant_int2t(e))
  {
    const BigInt &v = to_constant_int2t(e).value;
    if (int_mode)
      return "(" + integer2string(v) + " : Int)";
    const std::string w = std::to_string(t->get_width());
    if (v.is_negative())
      return "(BitVec.ofInt " + w + " (" + integer2string(v) + "))";
    return "(" + integer2string(v) + "#" + w + ")";
  }
  if (is_add2t(e))
    return bin(to_add2t(e).side_1, "+", to_add2t(e).side_2);
  if (is_sub2t(e))
    return bin(to_sub2t(e).side_1, "-", to_sub2t(e).side_2);
  if (is_mul2t(e))
    return bin(to_mul2t(e).side_1, "*", to_mul2t(e).side_2);
  if ((is_div2t(e) || is_modulus2t(e)) && int_mode)
  {
    // --ir truncates signed / and % (mk_truncated_div/rem) but keeps unsigned
    // ones Euclidean (smt_solver.cpp): Int.tdiv/tmod and Lean's / and %. SMT
    // leaves a zero divisor unspecified, so it must be ruled out by ESBMC.
    const expr2tc &b = *e->get_sub_expr(1);
    BigInt c;
    bool constant = constant_value(b, c);
    if (!div_checked && !(constant && !c.is_zero()))
      unsupported(
        "division by a variable under --ir with --no-div-by-zero-check");
    if (!is_signedbv_type(t))
      return bin(*e->get_sub_expr(0), is_div2t(e) ? "/" : "%", b);
    if (constant && !c.is_zero())
      divisor_signs[integer2string(c)] = c.is_negative() ? -1 : 1;
    return call(is_div2t(e) ? "Int.tdiv" : "Int.tmod", *e->get_sub_expr(0), b);
  }
  if (
    int_mode && (is_bitnot2t(e) || is_bitand2t(e) || is_bitor2t(e) ||
                 is_bitxor2t(e) || is_shl2t(e) || is_lshr2t(e) ||
                 is_ashr2t(e) || is_overflow2t(e) || is_overflow_neg2t(e)))
    unsupported(get_expr_id(e) + " under --ir (unbounded integers)");
  // ESBMC lowers `/` to SMT-LIB bvudiv/bvsdiv (smt_solver.cpp); Lean's
  // smtUDiv/smtSDiv agree with them, including on a zero divisor.
  if (is_div2t(e))
    return call(
      is_signedbv_type(t) ? "BitVec.smtSDiv" : "BitVec.smtUDiv",
      to_div2t(e).side_1,
      to_div2t(e).side_2);
  if (is_modulus2t(e))
    return is_signedbv_type(t)
             ? call(
                 "BitVec.srem", to_modulus2t(e).side_1, to_modulus2t(e).side_2)
             : bin(to_modulus2t(e).side_1, "%", to_modulus2t(e).side_2);
  if (is_neg2t(e))
    return "(-" + value(to_neg2t(e).value) + ")";
  if (is_bitnot2t(e))
    return "(~~~" + value(to_bitnot2t(e).value) + ")";
  if (is_bitand2t(e))
    return bin(to_bitand2t(e).side_1, "&&&", to_bitand2t(e).side_2);
  if (is_bitor2t(e))
    return bin(to_bitor2t(e).side_1, "|||", to_bitor2t(e).side_2);
  if (is_bitxor2t(e))
    return bin(to_bitxor2t(e).side_1, "^^^", to_bitxor2t(e).side_2);
  if (is_shl2t(e))
    return bin(to_shl2t(e).side_1, "<<<", to_shl2t(e).side_2);
  if (is_lshr2t(e))
    return bin(to_lshr2t(e).side_1, ">>>", to_lshr2t(e).side_2);
  if (is_ashr2t(e))
    return call(
      "BitVec.sshiftRight'", to_ashr2t(e).side_1, to_ashr2t(e).side_2);
  if (is_if2t(e))
  {
    const if2t &i = to_if2t(e);
    return "(if " + value(i.cond) + " then " + value(i.true_value) + " else " +
           value(i.false_value) + ")";
  }
  if (is_typecast2t(e))
  {
    const expr2tc &from = to_typecast2t(e).from;
    if (is_bool_type(t) && is_bitvector(from->type))
      return "(" + value(from) + " != " + zero(from->type) + ")";
    if (is_bitvector(t) && is_bool_type(from->type))
    {
      const std::string one =
        int_mode ? "(1 : Int)" : "1#" + std::to_string(t->get_width());
      return "(if " + value(from) + " then " + one + " else " + zero(t) + ")";
    }
    if (is_bitvector(t) && is_bitvector(from->type))
    {
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
    unsupported(
      "cast from " + get_type_id(from->type) + " to " + get_type_id(t));
  }

  auto relation = [&](const expr2tc &a, const expr2tc &b, bool strict) {
    if (int_mode)
      return "(decide (" + value(a) + (strict ? " < " : " ≤ ") + value(b) +
             "))";
    const char *fn = is_signedbv_type(a->type)
                       ? (strict ? "BitVec.slt" : "BitVec.sle")
                       : (strict ? "BitVec.ult" : "BitVec.ule");
    return call(fn, a, b);
  };
  if (is_equality2t(e))
    return bin(to_equality2t(e).side_1, "==", to_equality2t(e).side_2);
  if (is_notequal2t(e))
    return bin(to_notequal2t(e).side_1, "!=", to_notequal2t(e).side_2);
  if (is_lessthan2t(e))
    return relation(to_lessthan2t(e).side_1, to_lessthan2t(e).side_2, true);
  if (is_lessthanequal2t(e))
    return relation(
      to_lessthanequal2t(e).side_1, to_lessthanequal2t(e).side_2, false);
  if (is_greaterthan2t(e))
    return relation(
      to_greaterthan2t(e).side_2, to_greaterthan2t(e).side_1, true);
  if (is_greaterthanequal2t(e))
    return relation(
      to_greaterthanequal2t(e).side_2, to_greaterthanequal2t(e).side_1, false);
  if (is_not2t(e))
    return "(!" + value(to_not2t(e).value) + ")";
  if (is_and2t(e))
    return bin(to_and2t(e).side_1, "&&", to_and2t(e).side_2);
  if (is_or2t(e))
    return bin(to_or2t(e).side_1, "||", to_or2t(e).side_2);
  if (is_xor2t(e))
    return bin(to_xor2t(e).side_1, "!=", to_xor2t(e).side_2);
  if (is_overflow2t(e))
  {
    const expr2tc &op = to_overflow2t(e).operand;
    auto sides =
      [&](const char *u, const char *s, const expr2tc &a, const expr2tc &b) {
        return call(is_signedbv_type(a->type) ? s : u, a, b);
      };
    if (is_add2t(op))
      return sides(
        "BitVec.uaddOverflow",
        "BitVec.saddOverflow",
        to_add2t(op).side_1,
        to_add2t(op).side_2);
    if (is_sub2t(op))
      return sides(
        "BitVec.usubOverflow",
        "BitVec.ssubOverflow",
        to_sub2t(op).side_1,
        to_sub2t(op).side_2);
    if (is_mul2t(op))
      return sides(
        "BitVec.umulOverflow",
        "BitVec.smulOverflow",
        to_mul2t(op).side_1,
        to_mul2t(op).side_2);
    if (is_div2t(op) || is_modulus2t(op))
    {
      // As smt_overflow.cpp: signed INT_MIN / -1, unsigned a zero divisor.
      const expr2tc &a = *op->get_sub_expr(0), &b = *op->get_sub_expr(1);
      if (is_signedbv_type(a->type))
        return call("BitVec.sdivOverflow", a, b);
      return "(" + value(b) + " == 0#" + std::to_string(b->type->get_width()) +
             ")";
    }
    unsupported("overflow check on " + get_expr_id(op));
  }
  if (is_overflow_neg2t(e))
  {
    const expr2tc &op = to_overflow_neg2t(e).operand;
    const std::string w = std::to_string(op->type->get_width());
    return is_signedbv_type(op->type)
             ? "(" + value(op) + " == BitVec.intMin " + w + ")"
             : "(" + value(op) + " != 0#" + w + ")";
  }
  if (is_forall2t(e) || is_exists2t(e))
    unsupported(
      "quantifier used as a value (only guards of ASSERT, ASSUME and GOTO)");
  unsupported("expression " + get_expr_id(e));
}

std::string goto2leant::prop(const expr2tc &raw)
{
  expr2tc e = strip_bool_roundtrip(raw);
  auto bin = [this](const expr2tc &a, const char *op, const expr2tc &b) {
    return "(" + prop(a) + " " + op + " " + prop(b) + ")";
  };
  auto cmp = [this](const expr2tc &a, const char *op, const expr2tc &b) {
    return "(" + value(a) + " " + op + " " + value(b) + ")";
  };
  auto signed_cmp = [this](const char *fn, const expr2tc &a, const expr2tc &b) {
    return "(" + std::string(fn) + " " + value(a) + " " + value(b) + " = true)";
  };

  if (is_constant_bool2t(e))
    return to_constant_bool2t(e).value ? "True" : "False";
  if (is_not2t(e))
  {
    expr2tc inner = strip_bool_roundtrip(to_not2t(e).value);
    if (is_not2t(inner))
      return prop(to_not2t(inner).value);
    return "¬" + prop(inner);
  }
  if (is_and2t(e))
    return bin(to_and2t(e).side_1, "∧", to_and2t(e).side_2);
  if (is_or2t(e))
    return bin(to_or2t(e).side_1, "∨", to_or2t(e).side_2);
  if (is_implies2t(e))
    return bin(to_implies2t(e).side_1, "→", to_implies2t(e).side_2);
  if (is_equality2t(e))
    return cmp(to_equality2t(e).side_1, "=", to_equality2t(e).side_2);
  if (is_notequal2t(e))
    return cmp(to_notequal2t(e).side_1, "≠", to_notequal2t(e).side_2);

  if (
    is_lessthan2t(e) || is_lessthanequal2t(e) || is_greaterthan2t(e) ||
    is_greaterthanequal2t(e))
  {
    const expr2tc &a = *e->get_sub_expr(0), &b = *e->get_sub_expr(1);
    bool strict = is_lessthan2t(e) || is_greaterthan2t(e);
    bool swap = is_greaterthan2t(e) || is_greaterthanequal2t(e);
    const expr2tc &l = swap ? b : a, &r = swap ? a : b;
    if (is_signedbv_type(a->type) && !int_mode)
      return signed_cmp(strict ? "BitVec.slt" : "BitVec.sle", l, r);
    return cmp(l, strict ? "<" : "≤", r);
  }

  if (is_forall2t(e) || is_exists2t(e))
  {
    has_quantifiers = true;
    const expr2tc &binder = quantifier_binder(
      is_forall2t(e) ? to_forall2t(e).side_1 : to_exists2t(e).side_1);
    const expr2tc &body =
      is_forall2t(e) ? to_forall2t(e).side_2 : to_exists2t(e).side_2;
    if (!is_symbol2t(binder))
      unsupported("quantifier binder " + get_expr_id(binder));
    qbinders.push_back(to_symbol2t(binder).thename);
    std::string inner = prop(body);
    qbinders.pop_back();
    return "(" + std::string(is_forall2t(e) ? "∀ " : "∃ ") +
           var_name(to_symbol2t(binder).thename) + " : " + type(binder->type) +
           ", " + inner + ")";
  }

  if (is_typecast2t(e) && is_bitvector(to_typecast2t(e).from->type))
  {
    const expr2tc &from = to_typecast2t(e).from;
    return "(" + value(from) + " ≠ " + zero(from->type) + ")";
  }

  assert(is_bool_type(e->type));
  return "(" + value(e) + " = true)";
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
  // Expressions are translated against this path's bindings, before recursing.
  auto val = [&](const expr2tc &e) {
    in_scope = &scope;
    return value(e);
  };
  auto pr = [&](const expr2tc &e) {
    in_scope = &scope;
    return prop(e);
  };
  auto bind = [&](const expr2tc &sym) {
    const irep_idt &id = to_symbol2t(sym).thename;
    for (const vart &v : scope)
      if (v.id == id)
        return v.name;
    scope.push_back(vart{var_name(id), sym->type, id});
    return scope.back().name;
  };

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
    val(to_code_expression2t(I.code).operand);
    return next();

  case DECL:
  {
    const code_decl2t &d = to_code_decl2t(I.code);
    vart v{var_name(d.value), d.type, d.value};
    std::string t = type(v.type);
    scope.erase(
      std::remove_if(
        scope.begin(),
        scope.end(),
        [&](const vart &w) { return w.id == v.id; }),
      scope.end());
    scope.push_back(v);
    // goto_convert emits declarations without initialisers (the value
    // follows as an assignment), so an unassigned declaration is nondet.
    assert(is_nil_expr(d.init));
    std::string head;
    if (!assigned_before_use(pc + 1, end, d.value))
      head = p + "∀ " + v.name + " : " + t + ",\n";
    return head + trans(pc + 1, end, scope, loop, on_end, ind);
  }

  case ASSIGN:
  {
    const code_assign2t &a = to_code_assign2t(I.code);
    if (!is_symbol2t(a.target))
      unsupported("assignment to " + get_expr_id(a.target), I.location);
    bool nondet = is_sideeffect2t(a.source) && to_sideeffect2t(a.source).kind ==
                                                 sideeffect_allockind::nondet;
    if (!nondet)
    {
      std::string rhs = val(a.source);
      return p + "let " + bind(a.target) + " := " + rhs + ";\n" + next();
    }
    std::string x = bind(a.target);

    // `tmp = NONDET; v = tmp;` binds `v` directly, so Lean's counterexamples
    // name the program variable rather than ESBMC's temporary.
    if (pc + 1 < end && cur->insts[pc + 1]->type == ASSIGN)
    {
      const code_assign2t &b = to_code_assign2t(cur->insts[pc + 1]->code);
      if (
        is_symbol2t(b.target) && b.source == a.target &&
        b.target->type == a.target->type)
      {
        std::string v = bind(b.target);
        return p + "∀ " + v + " : " + type(b.target->type) + ",\n" + p +
               "let " + x + " := " + v + ";\n" +
               trans(pc + 2, end, scope, loop, on_end, ind);
      }
    }
    return p + "∀ " + x + " : " + type(a.target->type) + ",\n" + next();
  }

  case ASSUME:
  {
    std::string g = pr(I.guard);
    return p + g + " →\n" + next();
  }

  case ASSERT:
  {
    std::string g = pr(I.guard);
    return p + "-- " + I.location.comment().as_string() + "\n" + p + g +
           " ∧ (\n" + next() + ")";
  }

  case FUNCTION_CALL:
  {
    const code_function_call2t &c = to_code_function_call2t(I.code);
    if (!is_symbol2t(c.function))
      unsupported("indirect function call", I.location);
    const std::string callee = to_symbol2t(c.function).thename.as_string();
    if (!is_nil_expr(c.ret) && !is_symbol2t(c.ret))
      unsupported("call result stored in " + get_expr_id(c.ret), I.location);
    std::string args;
    for (const expr2tc &arg : c.operands)
      args += " " + val(arg); // also rejects what symex would check here
    auto it = fn_names.find(callee);
    if (it == fn_names.end())
    {
      // ESBMC intrinsics have semantics of their own (e.g. unreachability).
      const std::string name = shorthand(callee);
      if (name.rfind("__ESBMC", 0) == 0 || name.rfind("__VERIFIER", 0) == 0)
        unsupported("call to intrinsic " + name, I.location);
      // No body: the only effect modelled is a nondet result.
      if (is_nil_expr(c.ret))
        return p + "-- call to " + name + " (no body): ignored\n" + next();
      std::string r = bind(c.ret);
      return p + "-- call to " + name + " (no body): nondet result\n" + p +
             "∀ " + r + " : " + type(c.ret->type) + ",\n" + next();
    }
    // As with nested loops, a fixpoint is not monotone through a callee.
    if (loop)
      unsupported(
        "call to " + shorthand(callee) + " inside a loop", I.location);
    std::string binder = is_nil_expr(c.ret) ? "_" : bind(c.ret);
    return p + it->second + args + " (fun " + binder + " =>\n" + next() + ")";
  }

  case RETURN:
  {
    const code_return2t &r = to_code_return2t(I.code);
    return p + ret_cont(loop) +
           (is_nil_expr(r.operand) ? " ()" : " " + val(r.operand));
  }

  case END_FUNCTION:
    if (is_empty_type(cur->ret_type))
      return p + ret_cont(loop) + " ()";
    return p + "∀ result : " + type(cur->ret_type) + ", " + ret_cont(loop) +
           " result";

  case GOTO:
  {
    std::size_t t = cur->index.at(&*I.targets.front());
    if (is_true(I.guard))
      return jump(t, end, scope, loop, on_end, ind);
    std::string g = pr(I.guard);

    // A jump to the enclosing loop's head or exit, or out of this region.
    if ((loop && (t == loop->head || t == loop->exit)) || t >= end)
      return p + "(" + g + " →\n" + jump(t, end, scope, loop, on_end, ind + 1) +
             ") ∧\n" + p + "(" + negate(g) + " →\n" + next() + ")";
    if (t <= pc)
      unsupported("backward conditional jump", I.location);

    // if (!g) { then } [else { els }]: the else branch, when present, is the
    // range [t, join) skipped by the unconditional jump that ends `then`.
    std::size_t join = t, then_end = t;
    const instructiont &before = *cur->insts[t - 1];
    if (t - 1 > pc && before.is_goto() && is_true(before.guard))
    {
      std::size_t e = cur->index.at(&*before.targets.front());
      if (e > t && e <= end)
        join = e, then_end = t - 1;
    }
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

  default:
    unsupported(
      "instruction type " + std::to_string(static_cast<int>(I.type)),
      I.location);
  }
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
    const instructiont &i = *cur->insts[pc];
    if (i.is_decl() || i.type == DEAD || i.type == SKIP || i.type == LOCATION)
      continue;
    if ((i.is_assert() || i.is_assume()) && !mentions(i.guard, id))
      continue;
    if (i.is_assign())
    {
      const code_assign2t &a = to_code_assign2t(i.code);
      if (mentions(a.source, id))
        return false;
      if (is_symbol2t(a.target) && to_symbol2t(a.target).thename == id)
        return true;
      continue;
    }
    if (i.is_function_call())
    {
      const code_function_call2t &c = to_code_function_call2t(i.code);
      for (const expr2tc &arg : c.operands)
        if (mentions(arg, id))
          return false;
      if (is_symbol2t(c.ret) && to_symbol2t(c.ret).thename == id)
        return true;
      continue;
    }
    return false;
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

std::string goto2leant::translate()
{
  for (const auto &[id, fn] : goto_functions.function_map)
    if (fn.body_available && source_name(id) == entry && entry_id.empty())
      entry_id = id.as_string();
  if (entry_id.empty())
    throw goto2lean_unsupported(
      "goto2lean: entry function " + entry + " not found");

  // ESBMC runs start-up code before the entry (C++ static constructors,
  // Python module code); its assertions are not part of this translation.
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
  if (int_mode && simplifier_on)
    unsupported(
      "--ir without --no-simplify: ESBMC's simplifier folds constants with "
      "fixed-width wraparound, which Lean's Int does not");

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
      << entry_fn << " (fun _ => True) := by\n";
  if (has_loops || has_quantifiers)
  {
    out << "  -- Not closed automatically: "
        << (has_loops
              ? "each loop needs an invariant (apply Prog.<loop>.coinduct)"
              : "bv_decide only handles quantifier-free goals")
        << ".\n  sorry\n";
  }
  else
  {
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
  }
  return out.str();
}
