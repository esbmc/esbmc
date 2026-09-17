#include <esbmc/invariant_mining.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>
#include <algorithm>
#include <map>

namespace
{
using valuest = std::map<irep_idt, BigInt>;

/// Fewest samples a relation must hold on to be guessed at all.
constexpr size_t kMinSamples = 3;

/// Variables considered per loop. Relations are guessed between pairs and
/// generalised over every unwritten variable, so the guesses grow with the
/// square of the first and linearly in the second.
constexpr size_t kMaxModifiedColumns = 8;
constexpr size_t kMaxUnmodifiedColumns = 4;

/// What the samples of one loop agree on: every variable they all have a
/// value for, with its values in sample order.
struct columnt
{
  expr2tc symbol;
  bool modified;
  std::vector<BigInt> values;
};

std::vector<columnt>
columns_of(const std::vector<const loop_head_samplet *> &samples)
{
  std::map<irep_idt, columnt> columns;
  for (const auto &v : samples.front()->values)
    columns[to_symbol2t(v.symbol).thename] = {v.symbol, v.modified, {}};

  for (const loop_head_samplet *s : samples)
  {
    std::map<irep_idt, BigInt> present;
    for (const auto &v : s->values)
      present[to_symbol2t(v.symbol).thename] = v.value;
    for (auto it = columns.begin(); it != columns.end();)
    {
      const auto found = present.find(it->first);
      if (found == present.end())
      {
        it = columns.erase(it);
        continue;
      }
      it->second.values.push_back(found->second);
      ++it;
    }
  }

  std::vector<columnt> out;
  size_t modified = 0, unmodified = 0;
  for (auto &[name, column] : columns)
  {
    size_t &taken = column.modified ? modified : unmodified;
    if (taken == (column.modified ? kMaxModifiedColumns : kMaxUnmodifiedColumns))
      continue;
    ++taken;
    out.push_back(std::move(column));
  }
  return out;
}

expr2tc wide(const expr2tc &e)
{
  return typecast2tc(get_int_type(64), e);
}

expr2tc wide_constant(const BigInt &v)
{
  return constant_int2tc(get_int_type(64), v);
}

/// `w == a * v + b` over the samples, with integer a and b, if it holds.
bool affine_relation(
  const columnt &v,
  const columnt &w,
  BigInt &a,
  BigInt &b)
{
  size_t first = 0, second = 1;
  while (second < v.values.size() && v.values[second] == v.values[first])
    ++second;
  if (second == v.values.size())
    return false;

  const BigInt dv = v.values[second] - v.values[first];
  const BigInt dw = w.values[second] - w.values[first];
  if (dw % dv != 0)
    return false;
  a = dw / dv;
  b = w.values[first] - a * v.values[first];

  for (size_t i = 0; i < v.values.size(); ++i)
    if (w.values[i] != a * v.values[i] + b)
      return false;
  return true;
}

void add_affine_relations(
  const std::vector<columnt> &columns,
  unsigned loop,
  std::vector<kind_candidatet> &out)
{
  for (const columnt &w : columns)
  {
    if (!w.modified)
      continue;
    for (const columnt &v : columns)
    {
      BigInt a, b;
      if (&v == &w || !v.modified || !affine_relation(v, w, a, b))
        continue;

      const expr2tc scaled = mul2tc(get_int_type(64), wide_constant(a), wide(v.symbol));
      out.push_back(
        {loop,
         equality2tc(wide(w.symbol), add2tc(get_int_type(64), scaled, wide_constant(b))),
         "mined"});

      // The offset may be a variable the loop leaves alone, which a single
      // execution cannot tell from the constant it held.
      for (const columnt &u : columns)
      {
        if (u.modified)
          continue;
        const BigInt &uv = u.values.front();
        out.push_back(
          {loop,
           equality2tc(
             wide(w.symbol),
             add2tc(
               get_int_type(64),
               add2tc(get_int_type(64), scaled, wide(u.symbol)),
               wide_constant(b - uv))),
           "mined"});
      }
    }
  }
}

void add_bounds_and_parity(
  const std::vector<columnt> &columns,
  unsigned loop,
  std::vector<kind_candidatet> &out)
{
  for (const columnt &c : columns)
  {
    if (!c.modified)
      continue;
    const auto [lo, hi] = std::minmax_element(c.values.begin(), c.values.end());
    out.push_back(
      {loop, greaterthanequal2tc(wide(c.symbol), wide_constant(*lo)), "mined"});
    out.push_back(
      {loop, lessthanequal2tc(wide(c.symbol), wide_constant(*hi)), "mined"});

    const BigInt parity = c.values.front() % 2;
    if (std::all_of(c.values.begin(), c.values.end(), [&](const BigInt &x) {
          return x % 2 == parity;
        }))
      out.push_back(
        {loop,
         equality2tc(
           modulus2tc(get_int_type(64), wide(c.symbol), wide_constant(2)),
           wide_constant(parity)),
         "mined"});
  }

  for (const columnt &v : columns)
    for (const columnt &w : columns)
    {
      if (&v >= &w || !v.modified || !w.modified)
        continue;
      std::vector<BigInt> diffs;
      for (size_t i = 0; i < v.values.size(); ++i)
        diffs.push_back(v.values[i] - w.values[i]);
      const auto [lo, hi] = std::minmax_element(diffs.begin(), diffs.end());
      const expr2tc diff = sub2tc(get_int_type(64), wide(v.symbol), wide(w.symbol));
      out.push_back({loop, greaterthanequal2tc(diff, wide_constant(*lo)), "mined"});
      out.push_back({loop, lessthanequal2tc(diff, wide_constant(*hi)), "mined"});
    }
}

/// @p e with every symbol replaced by its value, or nil when a symbol has none.
expr2tc substitute(const expr2tc &e, const valuest &values)
{
  if (is_symbol2t(e))
  {
    const auto found = values.find(to_symbol2t(e).thename);
    if (found == values.end() || !is_bv_type(e))
      return expr2tc();
    return constant_int2tc(e->type, found->second);
  }

  expr2tc out = e;
  bool known = true;
  out->Foreach_operand([&](expr2tc &op) {
    if (!known)
      return;
    op = substitute(op, values);
    known = !is_nil_expr(op);
  });
  return known ? out : expr2tc();
}
} // namespace

std::vector<kind_candidatet>
mine_candidates(const std::vector<loop_head_samplet> &samples)
{
  std::map<unsigned, std::vector<const loop_head_samplet *>> by_loop;
  for (const loop_head_samplet &s : samples)
    by_loop[s.loop].push_back(&s);

  std::vector<kind_candidatet> out;
  for (const auto &[loop, loop_samples] : by_loop)
  {
    if (loop_samples.size() < kMinSamples)
      continue;
    const std::vector<columnt> columns = columns_of(loop_samples);
    add_affine_relations(columns, loop, out);
    add_bounds_and_parity(columns, loop, out);
  }
  return out;
}

std::vector<kind_candidatet> separate_counterexample(
  const std::vector<loop_head_samplet> &samples,
  const loop_head_samplet &cti)
{
  std::vector<const loop_head_samplet *> loop_samples;
  for (const loop_head_samplet &s : samples)
    if (s.loop == cti.loop)
      loop_samples.push_back(&s);
  if (loop_samples.empty())
    return {};

  valuest at_cti;
  for (const auto &v : cti.values)
    at_cti[to_symbol2t(v.symbol).thename] = v.value;

  std::vector<kind_candidatet> out;
  auto separate = [&](
                    const expr2tc &term,
                    const std::vector<BigInt> &reachable,
                    const BigInt &value) {
    const auto [lo, hi] =
      std::minmax_element(reachable.begin(), reachable.end());
    if (value < *lo)
      out.push_back(
        {cti.loop, greaterthanequal2tc(term, wide_constant(*lo)), "cti"});
    if (value > *hi)
      out.push_back(
        {cti.loop, lessthanequal2tc(term, wide_constant(*hi)), "cti"});
  };

  const std::vector<columnt> columns = columns_of(loop_samples);
  for (const columnt &c : columns)
  {
    const auto value = at_cti.find(to_symbol2t(c.symbol).thename);
    if (c.modified && value != at_cti.end())
      separate(wide(c.symbol), c.values, value->second);
  }

  for (const columnt &v : columns)
    for (const columnt &w : columns)
    {
      const auto vv = at_cti.find(to_symbol2t(v.symbol).thename);
      const auto wv = at_cti.find(to_symbol2t(w.symbol).thename);
      if (
        &v >= &w || !v.modified || !w.modified || vv == at_cti.end() ||
        wv == at_cti.end())
        continue;
      std::vector<BigInt> diffs;
      for (size_t i = 0; i < v.values.size(); ++i)
        diffs.push_back(v.values[i] - w.values[i]);
      separate(
        sub2tc(get_int_type(64), wide(v.symbol), wide(w.symbol)),
        diffs,
        vv->second - wv->second);
    }
  return out;
}

bool refuted_by_samples(
  const kind_candidatet &candidate,
  const std::vector<loop_head_samplet> &samples)
{
  for (const loop_head_samplet &s : samples)
  {
    if (s.loop != candidate.loop)
      continue;
    valuest values;
    for (const auto &v : s.values)
      values[to_symbol2t(v.symbol).thename] = v.value;
    expr2tc evaluated = substitute(candidate.expr, values);
    if (is_nil_expr(evaluated))
      continue;
    simplify(evaluated);
    if (is_false(evaluated))
      return true;
  }
  return false;
}
