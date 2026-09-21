#pragma once

#include <solvers/smt/smt_solver.h>

#include <functional>

/// BTOR2 has no floating-point sort, so floats reaching the writer have to be
/// bit-blasted. ESBMC already owns that lowering in fp_convt, which builds
/// terms through an smt_solver_baset; this adapts that interface onto the
/// BTOR2 writer, emitting one line per operation instead of building a solver
/// term. Only term construction is supported — fp_convt never asks the solver
/// anything on that path (its one ctx->get_bv call is in model readback), so
/// the solving and model-readback members abort if they are ever reached.
class btor2_fp_convt : public smt_solver_baset
{
public:
  /// \p emit_op writes one BTOR2 line ("add", width, operands) and returns its
  /// node id; \p emit_const writes a constant of the given width.
  using emit_opt = std::function<
    unsigned(const std::string &, unsigned, const std::vector<unsigned> &)>;
  using emit_constt = std::function<unsigned(const BigInt &, unsigned)>;

  btor2_fp_convt(
    const namespacet &ns,
    const optionst &options,
    emit_opt emit_op,
    emit_constt emit_const);

  /// Wrap a BTOR2 node of the given width so fp_convt can consume it.
  smt_astt wrap(unsigned node, unsigned width);
  /// Same node, carrying a different sort. fp_convt re-tags a bit-vector as a
  /// float this way; the default smt_ast::with_sort mutates in place, which
  /// would corrupt every other holder of a hash-consed node.
  smt_astt with_sort_of(smt_astt a, smt_sortt s);
  /// The BTOR2 node behind a term fp_convt produced.
  static unsigned node_of(smt_astt a);
  /// The softfloat lowering this adapter drives.
  fp_convt &fp() const
  {
    return *fp_api;
  }

  const std::string solver_text() override
  {
    return "btor2";
  }

  // BTOR2 has bit-vector sorts only, so a float sort is its bit width and a
  // rounding mode is the three bits fp_convt encodes it in.
  smt_sortt mk_bool_sort() override;
  smt_sortt mk_bv_sort(std::size_t width) override;
  smt_sortt mk_bvfp_sort(std::size_t ew, std::size_t sw) override;
  smt_sortt mk_bvfp_rm_sort() override;

  smt_astt mk_smt_bv(const BigInt &theint, smt_sortt s) override;
  smt_astt mk_smt_bool(bool val) override;
  smt_astt mk_smt_symbol(const std::string &name, smt_sortt s) override;
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_concat(smt_astt a, smt_astt b) override;
  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f) override;
  smt_astt mk_extract(smt_astt a, unsigned int high, unsigned int low) override;

  smt_astt mk_and(smt_astt a, smt_astt b) override;
  smt_astt mk_or(smt_astt a, smt_astt b) override;
  smt_astt mk_xor(smt_astt a, smt_astt b) override;
  smt_astt mk_not(smt_astt a) override;
  smt_astt mk_eq(smt_astt a, smt_astt b) override;

  smt_astt mk_bvadd(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsub(smt_astt a, smt_astt b) override;
  smt_astt mk_bvmul(smt_astt a, smt_astt b) override;
  smt_astt mk_bvudiv(smt_astt a, smt_astt b) override;
  smt_astt mk_bvumod(smt_astt a, smt_astt b) override;
  smt_astt mk_bvneg(smt_astt a) override;
  smt_astt mk_bvnot(smt_astt a) override;
  smt_astt mk_bvand(smt_astt a, smt_astt b) override;
  smt_astt mk_bvor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvxor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvshl(smt_astt a, smt_astt b) override;
  smt_astt mk_bvlshr(smt_astt a, smt_astt b) override;
  smt_astt mk_bvashr(smt_astt a, smt_astt b) override;
  smt_astt mk_bvult(smt_astt a, smt_astt b) override;
  smt_astt mk_bvule(smt_astt a, smt_astt b) override;
  smt_astt mk_bvslt(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsle(smt_astt a, smt_astt b) override;

  // Term construction only: nothing below is reachable from fp_convt.
  void assert_ast(smt_astt) override;
  smt_resultt dec_solve() override;
  smt_astt mk_smt_int(const BigInt &) override;
  smt_astt mk_smt_real(const std::string &) override;
  tvt get_bool(smt_astt) override;
  BigInt get_bv(smt_astt, bool) override;

private:
  emit_opt emit_op;
  emit_constt emit_const;
  std::vector<std::unique_ptr<smt_ast>> owned;
  std::vector<std::unique_ptr<smt_sort>> owned_sorts;
  std::unordered_map<unsigned, smt_sortt> sort_cache;
  /// One node per distinct name, so a nondeterministic value fp_convt asks for
  /// twice is the same value, not two independent BTOR2 inputs.
  std::unordered_map<std::string, smt_astt> symbols;

  smt_sortt bv_sort(unsigned width);
  unsigned width_of_ast(smt_astt a) const;
  /// Hash-consing, as every real backend does: an operation on the same
  /// operands must return the *same* term object, or fp_convt's shared
  /// subterms turn into a tree and both its work and ours blow up.
  smt_astt node(
    const std::string &name,
    unsigned width,
    const std::vector<unsigned> &args);
  std::unordered_map<std::string, smt_astt> consed;
};
