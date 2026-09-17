#include <esbmc/ts/ts_btor2.h>

#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/arith/mp_arith.h>

#include <map>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace
{
unsigned width_of(const type2tc &t)
{
  if (is_bool_type(t))
    return 1;
  if (is_signedbv_type(t) || is_unsignedbv_type(t))
    return t->get_width();
  throw std::runtime_error("unsupported type " + get_type_id(t));
}

std::string sanitize(const std::string &name)
{
  std::string out = "v_";
  for (char c : name)
    out += isalnum(static_cast<unsigned char>(c)) ? c : '_';
  return out;
}

class btor2_writert
{
public:
  btor2_writert(const transition_systemt &ts, std::ostream &out)
    : ts(ts), out(out)
  {
  }

  void write();

private:
  const transition_systemt &ts;
  std::ostream &out;
  unsigned last = 0;
  std::map<unsigned, unsigned> sorts;
  std::unordered_map<expr2tc, unsigned, irep2_hash> leaves;
  std::unordered_map<expr2tc, expr2tc, irep2_hash> defs;
  std::unordered_map<const expr2t *, unsigned> memo;

  unsigned emit(const std::string &line)
  {
    out << ++last << ' ' << line << '\n';
    return last;
  }

  unsigned sort(unsigned width)
  {
    auto it = sorts.find(width);
    if (it != sorts.end())
      return it->second;
    return sorts[width] = emit("sort bitvec " + std::to_string(width));
  }

  unsigned
  op(const std::string &name, unsigned width, std::vector<unsigned> args)
  {
    std::string line = name + ' ' + std::to_string(sort(width));
    for (unsigned a : args)
      line += ' ' + std::to_string(a);
    return emit(line);
  }

  unsigned bool_const(bool v)
  {
    return op(v ? "one" : "zero", 1, {});
  }

  unsigned conjunction(const std::vector<unsigned> &args)
  {
    if (args.empty())
      return bool_const(true);
    unsigned acc = args[0];
    for (size_t i = 1; i < args.size(); i++)
      acc = op("and", 1, {acc, args[i]});
    return acc;
  }

  unsigned disjunction(const std::vector<unsigned> &args)
  {
    if (args.empty())
      return bool_const(false);
    unsigned acc = args[0];
    for (size_t i = 1; i < args.size(); i++)
      acc = op("or", 1, {acc, args[i]});
    return acc;
  }

  const expr2tc &resolve(const expr2tc &e) const;
  void collect_leaves(
    const expr2tc &e,
    std::unordered_set<expr2tc, irep2_hash> &leaves_out,
    std::unordered_set<const expr2t *> &visited) const;
  unsigned cast(unsigned node, const type2tc &from, const type2tc &to);
  unsigned convert_as(const expr2tc &e, const type2tc &t)
  {
    return cast(convert(e), e->type, t);
  }
  unsigned convert(const expr2tc &e);
  unsigned convert_expr(const expr2tc &e);
};

const expr2tc &btor2_writert::resolve(const expr2tc &e) const
{
  const expr2tc *cur = &e;
  while (is_symbol2t(*cur))
  {
    auto d = defs.find(*cur);
    if (d == defs.end())
      break;
    cur = &d->second;
  }
  return *cur;
}

void btor2_writert::collect_leaves(
  const expr2tc &e,
  std::unordered_set<expr2tc, irep2_hash> &leaves_out,
  std::unordered_set<const expr2t *> &visited) const
{
  if (is_nil_expr(e) || !visited.insert(e.get()).second)
    return;
  if (is_symbol2t(e))
  {
    auto d = defs.find(e);
    if (d == defs.end())
      leaves_out.insert(e);
    else
      collect_leaves(d->second, leaves_out, visited);
    return;
  }
  e->foreach_operand([&](const expr2tc &op) {
    collect_leaves(op, leaves_out, visited);
  });
}

unsigned
btor2_writert::cast(unsigned node, const type2tc &from, const type2tc &to)
{
  const unsigned fw = width_of(from), tw = width_of(to);
  if (is_bool_type(to))
    return is_bool_type(from)
             ? node
             : emit(
                 "neq " + std::to_string(sort(1)) + ' ' + std::to_string(node) +
                 ' ' + std::to_string(op("zero", fw, {})));
  if (tw == fw)
    return node;
  if (tw < fw)
    return emit(
      "slice " + std::to_string(sort(tw)) + ' ' + std::to_string(node) + ' ' +
      std::to_string(tw - 1) + " 0");
  return emit(
    std::string(is_signedbv_type(from) ? "sext " : "uext ") +
    std::to_string(sort(tw)) + ' ' + std::to_string(node) + ' ' +
    std::to_string(tw - fw));
}

unsigned btor2_writert::convert(const expr2tc &e)
{
  auto m = memo.find(e.get());
  if (m != memo.end())
    return m->second;
  unsigned id = convert_expr(e);
  memo.emplace(e.get(), id);
  return id;
}

unsigned btor2_writert::convert_expr(const expr2tc &e)
{
  const unsigned w = width_of(e->type);
  auto side = [&](const expr2tc &x, const type2tc &t) {
    return convert_as(x, t);
  };
  auto binary = [&](const char *name, const expr2tc &a, const expr2tc &b) {
    return op(name, w, {side(a, e->type), side(b, e->type)});
  };
  auto relation = [&](const char *s, const char *u, const expr2tc &a, const expr2tc &b) {
    return op(
      is_signedbv_type(a->type) ? s : u, 1, {convert(a), side(b, a->type)});
  };

  switch (e->expr_id)
  {
  case expr2t::symbol_id:
  {
    auto l = leaves.find(e);
    if (l != leaves.end())
      return l->second;
    auto d = defs.find(e);
    if (d != defs.end())
      return convert(d->second);
    unsigned id;
    if (ts.step_local.count(e))
      id = op("input", w, {});
    else
    {
      // Fixed before the loop: free initially, then unchanged.
      id = op("state", w, {});
      op("next", w, {id, id});
    }
    leaves.emplace(e, id);
    return id;
  }
  case expr2t::constant_int_id:
    return emit(
      "const " + std::to_string(sort(w)) + ' ' +
      integer2binary(to_constant_int2t(e).value, w));
  case expr2t::constant_bool_id:
    return bool_const(to_constant_bool2t(e).value);
  case expr2t::typecast_id:
    return side(to_typecast2t(e).from, e->type);
  case expr2t::if_id:
  {
    const if2t &i = to_if2t(e);
    return op(
      "ite",
      w,
      {side(i.cond, get_bool_type()),
       side(i.true_value, e->type),
       side(i.false_value, e->type)});
  }
  case expr2t::not_id:
    return op("not", 1, {side(to_not2t(e).value, get_bool_type())});
  case expr2t::and_id:
    return binary("and", to_and2t(e).side_1, to_and2t(e).side_2);
  case expr2t::or_id:
    return binary("or", to_or2t(e).side_1, to_or2t(e).side_2);
  case expr2t::xor_id:
    return binary("xor", to_xor2t(e).side_1, to_xor2t(e).side_2);
  case expr2t::implies_id:
    return binary("implies", to_implies2t(e).side_1, to_implies2t(e).side_2);
  case expr2t::equality_id:
  {
    const equality2t &eq = to_equality2t(e);
    return op("eq", 1, {convert(eq.side_1), side(eq.side_2, eq.side_1->type)});
  }
  case expr2t::notequal_id:
  {
    const notequal2t &ne = to_notequal2t(e);
    return op("neq", 1, {convert(ne.side_1), side(ne.side_2, ne.side_1->type)});
  }
  case expr2t::lessthan_id:
    return relation(
      "slt", "ult", to_lessthan2t(e).side_1, to_lessthan2t(e).side_2);
  case expr2t::lessthanequal_id:
    return relation(
      "slte",
      "ulte",
      to_lessthanequal2t(e).side_1,
      to_lessthanequal2t(e).side_2);
  case expr2t::greaterthan_id:
    return relation(
      "sgt", "ugt", to_greaterthan2t(e).side_1, to_greaterthan2t(e).side_2);
  case expr2t::greaterthanequal_id:
    return relation(
      "sgte",
      "ugte",
      to_greaterthanequal2t(e).side_1,
      to_greaterthanequal2t(e).side_2);
  case expr2t::add_id:
    return binary("add", to_add2t(e).side_1, to_add2t(e).side_2);
  case expr2t::sub_id:
    return binary("sub", to_sub2t(e).side_1, to_sub2t(e).side_2);
  case expr2t::mul_id:
    return binary("mul", to_mul2t(e).side_1, to_mul2t(e).side_2);
  case expr2t::div_id:
  {
    const div2t &d = to_div2t(e);
    const bool u =
      is_unsignedbv_type(d.side_1) && is_unsignedbv_type(d.side_2);
    return binary(u ? "udiv" : "sdiv", d.side_1, d.side_2);
  }
  case expr2t::modulus_id:
  {
    const modulus2t &m = to_modulus2t(e);
    const bool u =
      is_unsignedbv_type(m.side_1) && is_unsignedbv_type(m.side_2);
    return binary(u ? "urem" : "srem", m.side_1, m.side_2);
  }
  case expr2t::neg_id:
    return op("neg", w, {side(to_neg2t(e).value, e->type)});
  case expr2t::abs_id:
  {
    const expr2tc &v = to_abs2t(e).value;
    unsigned x = side(v, e->type);
    if (is_unsignedbv_type(v))
      return x;
    unsigned nonneg = op("sgte", 1, {x, op("zero", w, {})});
    return op("ite", w, {nonneg, x, op("neg", w, {x})});
  }
  case expr2t::bitand_id:
    return binary("and", to_bitand2t(e).side_1, to_bitand2t(e).side_2);
  case expr2t::bitor_id:
    return binary("or", to_bitor2t(e).side_1, to_bitor2t(e).side_2);
  case expr2t::bitxor_id:
    return binary("xor", to_bitxor2t(e).side_1, to_bitxor2t(e).side_2);
  case expr2t::bitnot_id:
    return op("not", w, {side(to_bitnot2t(e).value, e->type)});
  // The shift distance takes the shifted operand's type, as smt_convt does.
  case expr2t::shl_id:
    return binary("sll", to_shl2t(e).side_1, to_shl2t(e).side_2);
  case expr2t::lshr_id:
    return binary("srl", to_lshr2t(e).side_1, to_lshr2t(e).side_2);
  case expr2t::ashr_id:
    return binary("sra", to_ashr2t(e).side_1, to_ashr2t(e).side_2);
  case expr2t::extract_id:
  {
    const extract2t &x = to_extract2t(e);
    return emit(
      "slice " + std::to_string(sort(w)) + ' ' + std::to_string(convert(x.from)) +
      ' ' + std::to_string(x.upper) + ' ' + std::to_string(x.lower));
  }
  case expr2t::concat_id:
    return op(
      "concat",
      w,
      {convert(to_concat2t(e).side_1), convert(to_concat2t(e).side_2)});
  default:
    throw std::runtime_error("unsupported expression " + get_expr_id(*e));
  }
}

void btor2_writert::write()
{
  for (const auto *d : {&ts.prefix_defs, &ts.body_defs})
    for (const expr2tc &eq : *d)
      defs.emplace(to_equality2t(eq).side_1, to_equality2t(eq).side_2);

  const size_t n = ts.state_pre.size();
  // btor2parser requires an init value to precede its state, so constant
  // initial values are emitted before any state is declared.
  std::vector<unsigned> init_value(n, 0);
  for (size_t i = 0; i < n; i++)
  {
    const expr2tc &v = resolve(ts.state_init[i]);
    if (is_constant_int2t(v) || is_constant_bool2t(v))
      init_value[i] = convert_as(v, ts.state_pre[i]->type);
  }
  const unsigned yes = bool_const(true), no = bool_const(false);

  std::vector<unsigned> state(n);
  for (size_t i = 0; i < n; i++)
  {
    state[i] = emit(
      "state " + std::to_string(sort(width_of(ts.state_pre[i]->type))) + ' ' +
      sanitize(ts.state_names[i]));
    leaves.emplace(ts.state_pre[i], state[i]);
    if (init_value[i])
      op("init", width_of(ts.state_pre[i]->type), {state[i], init_value[i]});
  }

  // `first` holds in step 0 only; `valid` holds while every earlier step took
  // the back edge with its assumptions satisfied. A plain BTOR2 constraint
  // would also demand the back edge in the violating step, and so lose
  // violations followed by a loop exit.
  unsigned first = op("state", 1, {}), valid = op("state", 1, {});
  op("init", 1, {first, yes});
  op("next", 1, {first, no});
  op("init", 1, {valid, yes});

  std::unordered_set<expr2tc, irep2_hash> used;
  std::unordered_set<const expr2t *> visited;
  auto use = [&](const expr2tc &e) { collect_leaves(e, used, visited); };
  for (const auto &e : ts.state_post)
    use(e);
  for (const auto *v : {&ts.body_assumes, &ts.prefix_assumes})
    for (const auto &e : *v)
      use(e);
  for (const auto *props : {&ts.bad, &ts.prefix_bad})
    for (const auto &p : *props)
      use(p.violated);
  use(ts.back_guard);
  std::unordered_map<expr2tc, unsigned, irep2_hash> init_uses;
  for (const auto &e : ts.state_init)
    init_uses[resolve(e)]++;

  std::vector<unsigned> start;
  for (const auto &e : ts.prefix_assumes)
    start.push_back(convert(e));
  for (size_t i = 0; i < n; i++)
  {
    const expr2tc &v = resolve(ts.state_init[i]);
    const bool free_init = is_symbol2t(v) && !ts.step_local.count(v) &&
                           !used.count(v) && init_uses[v] == 1;
    if (!init_value[i] && !free_init)
      start.push_back(
        op("eq", 1, {state[i], convert_as(v, ts.state_pre[i]->type)}));
  }
  unsigned started = op("implies", 1, {first, conjunction(start)});

  std::vector<unsigned> step{started};
  for (const auto &e : ts.body_assumes)
    step.push_back(convert(e));
  step.push_back(convert(ts.back_guard));
  op("next", 1, {valid, op("and", 1, {valid, conjunction(step)})});
  for (size_t i = 0; i < n; i++)
    op(
      "next",
      width_of(ts.state_pre[i]->type),
      {state[i], convert_as(ts.state_post[i], ts.state_pre[i]->type)});

  std::vector<unsigned> violated, prefix_violated;
  for (const auto &p : ts.bad)
    violated.push_back(convert(p.violated));
  for (const auto &p : ts.prefix_bad)
    prefix_violated.push_back(convert(p.violated));
  unsigned in_loop =
    conjunction({valid, started, disjunction(violated)});
  unsigned before_loop =
    conjunction({first, started, disjunction(prefix_violated)});
  emit("bad " + std::to_string(op("or", 1, {in_loop, before_loop})));
}
} // namespace

void write_btor2(const transition_systemt &ts, std::ostream &out)
{
  btor2_writert(ts, out).write();
}
