#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>

namespace
{
type2tc word_type()
{
  return get_uint_type(config.ansi_c.word_size);
}

expr2tc gen_const(unsigned v)
{
  return constant_int2tc(word_type(), BigInt(v));
}

// Build a left-leaning n-ary chain: ((((0 + 1) + 2) + 3) + ... + n).
// Exercises the operand-walk path in expr2t::simplify(): every node has
// two expr2tc operands, get_num_sub_exprs() == 2, get_sub_expr(i) is
// O(num_fields) = O(small constant) per call, so the per-node cost is
// dominated by the fold over fields rather than the field count itself.
expr2tc build_add_chain(unsigned depth)
{
  expr2tc acc = gen_const(0);
  for (unsigned i = 1; i <= depth; ++i)
    acc = add2tc(word_type(), acc, gen_const(i));
  return acc;
}

// Build a wide constant_array2t: one field that is std::vector<expr2tc>
// of `count` elements. This is the shape where get_sub_expr(idx) is
// genuinely O(idx) — every call re-walks the vector from 0 looking for
// the desired index — and get_num_sub_exprs() + indexed get_sub_expr
// becomes O(n^2) overall.
expr2tc build_constant_array(unsigned count)
{
  type2tc subtype = word_type();
  type2tc array_ty = array_type2tc(subtype, expr2tc(), true);
  std::vector<expr2tc> members;
  members.reserve(count);
  for (unsigned i = 0; i < count; ++i)
    members.push_back(gen_const(i));
  return constant_array2tc(array_ty, members);
}
} // namespace

TEST_CASE("irep2 microbench: chain construction", "[bench]")
{
  config.ansi_c.word_size = 32;

  BENCHMARK("build add-chain depth=64") { return build_add_chain(64); };
  BENCHMARK("build add-chain depth=256") { return build_add_chain(256); };
}

TEST_CASE("irep2 microbench: simplify operand walk", "[bench]")
{
  config.ansi_c.word_size = 32;

  // The simplify hot path: every internal call performs
  //   for (idx = 0; idx < get_num_sub_exprs(); ++idx) get_sub_expr(idx);
  // For an add-chain depth=N, the top node has 2 operands but the
  // recursive simplify() over the whole chain walks 2N operands; the
  // cost we want to track is the per-walk indexed access overhead.
  expr2tc chain64 = build_add_chain(64);
  expr2tc chain256 = build_add_chain(256);

  BENCHMARK("simplify add-chain depth=64") { return chain64->simplify(); };
  BENCHMARK("simplify add-chain depth=256") { return chain256->simplify(); };
}

TEST_CASE("irep2 microbench: indexed sub-expr access", "[bench]")
{
  config.ansi_c.word_size = 32;

  // Wide vector field: get_sub_expr(idx) walks the std::vector<expr2tc>
  // from index 0 every call. The full loop is therefore O(n^2) in n =
  // count. D4 collapses this to a single for_each_field pass.
  expr2tc arr256 = build_constant_array(256);
  expr2tc arr1024 = build_constant_array(1024);

  BENCHMARK("indexed get_sub_expr loop count=256")
  {
    size_t n = arr256->get_num_sub_exprs();
    const expr2tc *last = nullptr;
    for (size_t i = 0; i < n; ++i)
      last = arr256->get_sub_expr(i);
    return last;
  };
  BENCHMARK("indexed get_sub_expr loop count=1024")
  {
    size_t n = arr1024->get_num_sub_exprs();
    const expr2tc *last = nullptr;
    for (size_t i = 0; i < n; ++i)
      last = arr1024->get_sub_expr(i);
    return last;
  };

  // Single-pass Foreach_operand: control case. The cost of *visiting*
  // every operand once, with no indexed re-walks. Distance between this
  // and the indexed loop above is the headroom D4 can reclaim.
  BENCHMARK("foreach_operand single pass count=256")
  {
    unsigned ops = 0;
    arr256->foreach_operand([&ops](const expr2tc &) { ++ops; });
    return ops;
  };
  BENCHMARK("foreach_operand single pass count=1024")
  {
    unsigned ops = 0;
    arr1024->foreach_operand([&ops](const expr2tc &) { ++ops; });
    return ops;
  };
}

TEST_CASE("irep2 microbench: clone + Foreach_operand rewrite", "[bench]")
{
  config.ansi_c.word_size = 32;

  // The second half of the simplify hot path: when an operand changed,
  // simplify() does clone() (one allocation + per-field copy through
  // the intrusive refcount) followed by Foreach_operand to splice the
  // rewritten operand list back in. Measure both halves together.
  expr2tc chain64 = build_add_chain(64);

  BENCHMARK("clone + foreach_operand rewrite depth=64")
  {
    expr2tc copy = chain64->clone();
    unsigned ops = 0;
    copy->Foreach_operand([&ops](expr2tc &) { ++ops; });
    return ops;
  };
}

TEST_CASE("irep2 microbench: CRC cache", "[bench]")
{
  config.ansi_c.word_size = 32;

  // F2 turned crc() into an acquire/release atomic with sentinel-zero:
  // first call computes + publishes, every subsequent call is a single
  // acquire load. The two benchmarks pin both halves of that contract.
  expr2tc chain64 = build_add_chain(64);

  BENCHMARK("crc cold (fresh tree)")
  {
    expr2tc fresh = build_add_chain(64);
    return fresh->crc();
  };
  BENCHMARK("crc warm (cached)") { return chain64->crc(); };
}

TEST_CASE("irep2 microbench: is_/to_/try_to_ dispatch", "[bench]")
{
  config.ansi_c.word_size = 32;

  // The tag-check + downcast path. F4 made these macro-generated and
  // identical-shaped; this bench is a noise-floor anchor for D10 and
  // any future audit that changes the dispatch surface.
  expr2tc add = add2tc(word_type(), gen_const(1), gen_const(2));

  BENCHMARK("is_add2t + to_add2t round-trip")
  {
    if (!is_add2t(add))
      return static_cast<const expr2tc *>(nullptr);
    const add2t &a = to_add2t(add);
    return &a.side_1;
  };
  BENCHMARK("try_to_add2t round-trip")
  {
    const add2t *a = try_to_add2t(add);
    return a ? &a->side_1 : static_cast<const expr2tc *>(nullptr);
  };
}
