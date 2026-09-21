#include <esbmc/ts/btor2_fp.h>

#include <solvers/smt/fp/fp_conv.h>
#include <util/arith/mp_arith.h>

#include <cstdlib>
#include <stdexcept>

btor2_fp_convt::btor2_fp_convt(
  const namespacet &ns,
  const optionst &options,
  emit_opt emit_op,
  emit_constt emit_const)
  : smt_solver_baset(ns, options),
    emit_op(std::move(emit_op)),
    emit_const(std::move(emit_const))
{
  // The generic softfloat lowering, the same one solve.cpp installs when a
  // solver has no native floating point.
  set_fp_conv(new fp_convt(this));
}

smt_sortt btor2_fp_convt::bv_sort(unsigned width)
{
  auto it = sort_cache.find(width);
  if (it != sort_cache.end())
    return it->second;
  owned_sorts.push_back(std::make_unique<solver_smt_sort<unsigned>>(
    SMT_SORT_BV, width, width));
  return sort_cache[width] = owned_sorts.back().get();
}

smt_sortt btor2_fp_convt::mk_bool_sort()
{
  return bv_sort(1);
}

smt_sortt btor2_fp_convt::mk_bv_sort(std::size_t width)
{
  return bv_sort(width);
}

smt_sortt btor2_fp_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  // Same shape the other backends use: sign ++ exponent ++ significand, and
  // the sort carries the significand width because fp_convt reads it back.
  owned_sorts.push_back(std::make_unique<solver_smt_sort<unsigned>>(
    SMT_SORT_BVFP, ew + sw + 1, ew + sw + 1, sw + 1));
  return owned_sorts.back().get();
}

smt_sortt btor2_fp_convt::mk_bvfp_rm_sort()
{
  owned_sorts.push_back(
    std::make_unique<solver_smt_sort<unsigned>>(SMT_SORT_BVFP_RM, 3, 3));
  return owned_sorts.back().get();
}

namespace
{
/// smt_ast::with_sort re-tags a term by mutating its sort in place and handing
/// back the same pointer (smt_solver.cpp). Terms here are hash-consed, so that
/// would silently change the sort under every other holder of the node —
/// return a fresh wrapper over the same BTOR2 node instead.
class btor2_ast : public solver_smt_ast<unsigned>
{
public:
  using solver_smt_ast<unsigned>::solver_smt_ast;

  smt_astt with_sort(smt_solver_baset *ctx, smt_sortt s) const override
  {
    return static_cast<btor2_fp_convt *>(ctx)->with_sort_of(this, s);
  }
};
} // namespace

smt_astt btor2_fp_convt::wrap(unsigned node, unsigned width)
{
  owned.push_back(std::make_unique<btor2_ast>(this, node, bv_sort(width)));
  return owned.back().get();
}

smt_astt btor2_fp_convt::with_sort_of(smt_astt a, smt_sortt s)
{
  owned.push_back(std::make_unique<btor2_ast>(this, node_of(a), s));
  return owned.back().get();
}

unsigned btor2_fp_convt::node_of(smt_astt a)
{
  return static_cast<const solver_smt_ast<unsigned> *>(a)->a;
}

unsigned btor2_fp_convt::width_of_ast(smt_astt a) const
{
  const unsigned w = a->sort->get_data_width();
  return w ? w : 1;
}

smt_astt btor2_fp_convt::node(
  const std::string &name,
  unsigned width,
  const std::vector<unsigned> &args)
{
  std::string key = name + ':' + std::to_string(width);
  for (unsigned a : args)
    key += ':' + std::to_string(a);
  auto it = consed.find(key);
  if (it != consed.end())
    return it->second;
  return consed[key] = wrap(emit_op(name, width, args), width);
}

/// One BTOR2 line, taking the result width from the first operand unless the
/// caller knows better.
#define BIN(name, op_name)                                                     \
  smt_astt btor2_fp_convt::name(smt_astt a, smt_astt b)                        \
  {                                                                            \
    const unsigned w = width_of_ast(a);                                        \
    return node(op_name, w, {node_of(a), node_of(b)});                         \
  }

BIN(mk_bvadd, "add")
BIN(mk_bvsub, "sub")
BIN(mk_bvmul, "mul")
BIN(mk_bvudiv, "udiv")
BIN(mk_bvumod, "urem")
BIN(mk_bvand, "and")
BIN(mk_bvor, "or")
BIN(mk_bvxor, "xor")
BIN(mk_bvshl, "sll")
BIN(mk_bvlshr, "srl")
BIN(mk_bvashr, "sra")
BIN(mk_and, "and")
BIN(mk_or, "or")
BIN(mk_xor, "xor")
#undef BIN

/// A predicate: one BTOR2 line of width 1 whatever the operands' width.
#define PRED(name, op_name)                                                    \
  smt_astt btor2_fp_convt::name(smt_astt a, smt_astt b)                        \
  {                                                                            \
    return node(op_name, 1, {node_of(a), node_of(b)});                         \
  }

PRED(mk_bvult, "ult")
PRED(mk_bvule, "ulte")
PRED(mk_bvslt, "slt")
PRED(mk_bvsle, "slte")
PRED(mk_eq, "eq")
#undef PRED

smt_astt btor2_fp_convt::mk_bvneg(smt_astt a)
{
  const unsigned w = width_of_ast(a);
  return node("neg", w, {node_of(a)});
}

smt_astt btor2_fp_convt::mk_bvnot(smt_astt a)
{
  const unsigned w = width_of_ast(a);
  return node("not", w, {node_of(a)});
}

smt_astt btor2_fp_convt::mk_not(smt_astt a)
{
  const unsigned w = width_of_ast(a);
  return node("not", w, {node_of(a)});
}

smt_astt btor2_fp_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  const unsigned w = width_of_ast(t);
  return node("ite", w, {node_of(cond), node_of(t), node_of(f)});
}

smt_astt btor2_fp_convt::mk_concat(smt_astt a, smt_astt b)
{
  const unsigned w = width_of_ast(a) + width_of_ast(b);
  return node("concat", w, {node_of(a), node_of(b)});
}

smt_astt
btor2_fp_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  // high < low would wrap the width to ~2^32 and take the whole export with
  // it, so fail loudly rather than emit a nonsense sort.
  if (high < low || high >= width_of_ast(a))
    throw std::runtime_error(
      "bad slice [" + std::to_string(high) + ':' + std::to_string(low) +
      "] of a " + std::to_string(width_of_ast(a)) + "-bit term");
  const unsigned w = high - low + 1;
  return node("slice", w, {node_of(a), high, low});
}

smt_astt btor2_fp_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  const unsigned w = width_of_ast(a) + topwidth;
  return node("sext", w, {node_of(a), topwidth});
}

smt_astt btor2_fp_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  const unsigned w = width_of_ast(a) + topwidth;
  return node("uext", w, {node_of(a), topwidth});
}

smt_astt btor2_fp_convt::mk_smt_bv(const BigInt &theint, smt_sortt s)
{
  const unsigned w = s->get_data_width() ? s->get_data_width() : 1;
  std::string key = "const:" + std::to_string(w) + ':' + integer2string(theint);
  auto it = consed.find(key);
  if (it != consed.end())
    return it->second;
  return consed[key] = wrap(emit_const(theint, w), w);
}

smt_astt btor2_fp_convt::mk_smt_bool(bool val)
{
  return node(val ? "one" : "zero", 1, {});
}

smt_astt btor2_fp_convt::mk_smt_symbol(const std::string &name, smt_sortt s)
{
  auto it = symbols.find(name);
  if (it != symbols.end())
    return it->second;
  const unsigned w = s->get_data_width() ? s->get_data_width() : 1;
  return symbols[name] = wrap(emit_op("input", w, {}), w);
}

// fp_convt builds terms and never solves or reads a model, so nothing below is
// reachable from it; abort rather than return something wrong.
void btor2_fp_convt::assert_ast(smt_astt)
{
  abort();
}

smt_resultt btor2_fp_convt::dec_solve()
{
  abort();
}

smt_astt btor2_fp_convt::mk_smt_int(const BigInt &)
{
  abort();
}

smt_astt btor2_fp_convt::mk_smt_real(const std::string &)
{
  abort();
}

tvt btor2_fp_convt::get_bool(smt_astt)
{
  abort();
}

BigInt btor2_fp_convt::get_bv(smt_astt, bool)
{
  abort();
}
