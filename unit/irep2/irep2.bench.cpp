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

  BENCHMARK("build add-chain depth=64")
  {
    return build_add_chain(64);
  };
  BENCHMARK("build add-chain depth=256")
  {
    return build_add_chain(256);
  };
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

  BENCHMARK("simplify add-chain depth=64")
  {
    return chain64->simplify();
  };
  BENCHMARK("simplify add-chain depth=256")
  {
    return chain256->simplify();
  };

  // Repeat-simplify: call simplify() a second time on the result of
  // the first simplification. Anchors what re-simplification costs
  // today (no idempotency marker), in case a future change tries to
  // short-circuit this path.
  expr2tc once64 = chain64->simplify();
  if (is_nil_expr(once64))
    once64 = chain64;
  expr2tc once256 = chain256->simplify();
  if (is_nil_expr(once256))
    once256 = chain256;

  BENCHMARK("re-simplify add-chain depth=64")
  {
    return once64->simplify();
  };
  BENCHMARK("re-simplify add-chain depth=256")
  {
    return once256->simplify();
  };
}

TEST_CASE("irep2 microbench: indexed sub-expr access", "[bench]")
{
  config.ansi_c.word_size = 32;

  // Wide vector field: get_sub_expr(idx) walks the std::vector<expr2tc>
  // from index 0 every call. The full loop is therefore O(n^2) in n =
  // count. The single-pass operand walk in simplify() collapses this.
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
  // and the indexed loop above is the headroom the single-pass walk
  // reclaims.
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

  // crc() is an acquire/release atomic with sentinel-zero: first call
  // computes + publishes, every subsequent call is a single acquire
  // load. The two benchmarks pin both halves of that contract.
  expr2tc chain64 = build_add_chain(64);

  BENCHMARK("crc cold (fresh tree)")
  {
    expr2tc fresh = build_add_chain(64);
    return fresh->crc();
  };
  BENCHMARK("crc warm (cached)")
  {
    return chain64->crc();
  };
}

TEST_CASE("irep2 microbench: is_/to_/try_to_ dispatch", "[bench]")
{
  config.ansi_c.word_size = 32;

  // The tag-check + downcast path. These are macro-generated and
  // identical-shaped; this bench is a noise-floor anchor for any
  // future audit that changes the dispatch surface.
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

TEST_CASE("irep2 microbench: irep_idt-bearing CRC", "[bench]")
{
  config.ansi_c.word_size = 32;

  // symbol2t has an irep_idt "thename" field. The crc() implementation
  // uses irep_idt::hash() (the interned table index) rather than
  // hashing the underlying std::string char-by-char.
  expr2tc sym1 =
    symbol2tc(word_type(), "some_quite_long_symbol_name_to_punish_hashing_v1");
  expr2tc sym2 =
    symbol2tc(word_type(), "some_quite_long_symbol_name_to_punish_hashing_v2");

  BENCHMARK("crc symbol (cold, fresh tree)")
  {
    expr2tc fresh = symbol2tc(
      word_type(), "some_quite_long_symbol_name_to_punish_hashing_v3");
    return fresh->crc();
  };
  BENCHMARK("crc symbol pair (warm)")
  {
    return sym1->crc() ^ sym2->crc();
  };

  // struct_type2t carries a std::vector<irep_idt> of member names —
  // a second site that benefits from the same hashing approach.
  type2tc struct_ty = []() {
    std::vector<type2tc> mtypes{word_type(), word_type(), word_type()};
    std::vector<irep_idt> mnames{
      "field_alpha_with_some_length",
      "field_beta_with_some_length",
      "field_gamma_with_some_length"};
    return struct_type2tc(mtypes, mnames, mnames, "test_struct_for_bench");
  }();

  BENCHMARK("crc struct type cold")
  {
    std::vector<type2tc> mtypes{word_type(), word_type(), word_type()};
    std::vector<irep_idt> mnames{
      "field_alpha_with_some_length",
      "field_beta_with_some_length",
      "field_gamma_with_some_length"};
    type2tc fresh =
      struct_type2tc(mtypes, mnames, mnames, "test_struct_for_bench");
    return fresh->crc();
  };
  BENCHMARK("crc struct type warm")
  {
    return struct_ty->crc();
  };
}

TEST_CASE("irep2 microbench: array/vector type construction", "[bench]")
{
  config.ansi_c.word_size = 32;

  // array_type2t / vector_type2t constructors short-circuit when the
  // size expression is already a constant_int (the common case from
  // frontends), so simplify() is only invoked on non-canonical sizes.
  type2tc subtype = word_type();
  expr2tc lit_size = constant_int2tc(subtype, BigInt(64));

  BENCHMARK("array_type2t with constant_int size")
  {
    return array_type2tc(subtype, lit_size, false);
  };
  BENCHMARK("vector_type2t with constant_int size")
  {
    return vector_type2tc(subtype, lit_size);
  };

  // Non-trivial size: an add tree the constructor will still simplify.
  // Anchors that the fast-path doesn't break the folding contract.
  expr2tc add_size = add2tc(
    subtype,
    constant_int2tc(subtype, BigInt(30)),
    constant_int2tc(subtype, BigInt(34)));

  BENCHMARK("array_type2t with add() size (folds to constant)")
  {
    return array_type2tc(subtype, add_size, false);
  };
}

TEST_CASE("irep2 microbench: cmp / lt on real nodes", "[bench]")
{
  config.ansi_c.word_size = 32;

  // operator== and operator< dispatch through the irep_methods2 fold,
  // which calls do_type_cmp / do_type_lt for each field. The benches
  // below stress those field-type dispatches across the catalogue:
  //
  //  * bv_type2t       — unsigned int + type_ids
  //  * pointer_type2t  — type2tc + bool + type_ids
  //  * symbol2t        — irep_idt + enum (renaming_level) + several uint
  //  * struct_type2t   — std::vector<type2tc> + std::vector<irep_idt>
  //  * add2t           — expr2tc x2 + type2tc
  //
  // Identical (structurally-equal) pairs to exercise the equality-true
  // path; distinct pairs to exercise the early-exit-on-mismatch path.

  type2tc bv32_a = unsignedbv_type2tc(32);
  type2tc bv32_b = unsignedbv_type2tc(32);
  type2tc bv64 = unsignedbv_type2tc(64);

  BENCHMARK("operator==(bv32, bv32) — equal")
  {
    return bv32_a == bv32_b;
  };
  BENCHMARK("operator==(bv32, bv64) — width differs")
  {
    return bv32_a == bv64;
  };
  BENCHMARK("operator<(bv32, bv64)")
  {
    return bv32_a < bv64;
  };

  type2tc ptr_a = pointer_type2tc(bv32_a);
  type2tc ptr_b = pointer_type2tc(bv32_a);
  BENCHMARK("operator==(ptr, ptr) — equal")
  {
    return ptr_a == ptr_b;
  };

  expr2tc sym_a = symbol2tc(bv32_a, "some_quite_long_symbol_name");
  expr2tc sym_b = symbol2tc(bv32_a, "some_quite_long_symbol_name");
  expr2tc sym_c = symbol2tc(bv32_a, "different_symbol_name_here");
  BENCHMARK("operator==(symbol, symbol) — equal")
  {
    return sym_a == sym_b;
  };
  BENCHMARK("operator==(symbol, symbol) — name differs")
  {
    return sym_a == sym_c;
  };

  std::vector<type2tc> mtypes{bv32_a, bv32_a, bv32_a};
  std::vector<irep_idt> mnames{"a", "b", "c"};
  type2tc s_a = struct_type2tc(mtypes, mnames, mnames, "S");
  type2tc s_b = struct_type2tc(mtypes, mnames, mnames, "S");
  BENCHMARK("operator==(struct, struct) — equal")
  {
    return s_a == s_b;
  };

  expr2tc add_a = add2tc(bv32_a, gen_const(1), gen_const(2));
  expr2tc add_b = add2tc(bv32_a, gen_const(1), gen_const(2));
  BENCHMARK("operator==(add, add) — equal")
  {
    return add_a == add_b;
  };
  BENCHMARK("operator<(add, add)")
  {
    return add_a < add_b;
  };
}
