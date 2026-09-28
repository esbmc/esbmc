#ifndef _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H
#define _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H

#include <solvers/smtlib/smtlib_conv.h>

#include <map>
#include <optional>

/** Backend for NeuroSym, a neural-guided SMT solver (a GAN proposes candidate
 *  models, with a Z3 fallback preserving soundness and completeness). NeuroSym
 *  natively parses only the QF_BV and QF_LIA fragments of SMT-LIB2, and this
 *  backend drives it as a pure QF_BV solver. NeuroSym runs as a separate
 *  program that cannot be linked into another application, so this backend
 *  reuses the smtlib backend's SMT-LIB2 serializer to render the formula into
 *  a file and runs NeuroSym on it in one-shot batch mode (--neurosym-prog,
 *  "%f" is replaced by the file path).
 *
 *  NeuroSym has no native array, floating-point, or tuple support, so the
 *  factory leaves those capability interfaces unset and ESBMC's flatteners
 *  lower everything to pure QF_BV; the header emitted before the formula is
 *  overridden to (set-logic QF_BV) accordingly. Integer/real encoding (--ir)
 *  is rejected in solve.cpp because the flattened int-mode logic would be
 *  QF_AUFLIRA, which NeuroSym cannot parse.
 *
 *  On a sat verdict NeuroSym prints its own
 *  "(model (define-fun NAME () SORT VALUE) ...)" block, which dec_solve()
 *  parses into local_model, so a counterexample needs no second solve.
 *  Composite queries are evaluated from those leaves; anything unhandled
 *  still falls back to --neurosym-model-prog. */
class neurosym_convt : public smtlib_convt
{
public:
  neurosym_convt(const namespacet &ns, const optionst &options);
  ~neurosym_convt() override;

  smt_resultt dec_solve() override;
  const std::string solver_text() override;
  std::string dump_smt() override;

  bool has_model() const override
  {
    return !local_model.empty() || smtlib_convt::has_model();
  }

  tvt get_bool(smt_astt a) override;
  tvt l_get(smt_astt a) override;
  BigInt get_bv(smt_astt a, bool is_signed) override;

private:
  neurosym_convt(
    const namespacet &ns,
    const optionst &options,
    const std::string &formula_path);

  /** An absent block, or an entry this cannot read, leaves those variables
   *  to the --neurosym-model-prog fallback. */
  void parse_model_block(const std::string &output);

  /** Raw value text as NeuroSym printed it: a numeral, a #x literal, or
   *  "true"/"false" -- the forms interp_numeric() already reads. */
  std::map<std::string, std::string> local_model;

  /** Unsigned bit pattern for a declared symbol; nullopt on a miss or a
   *  form numeric_value() cannot read. */
  std::optional<BigInt> local_lookup(const std::string &symname) const;

  /** Evaluate a bit-vector expression from local_model's leaves. The result
   *  is always masked to the expression's width, unsigned; callers apply
   *  their own sign interpretation. nullopt on any unknown leaf or node
   *  kind, which sends the caller to --neurosym-model-prog. */
  std::optional<BigInt> local_eval_bv(smt_astt a) const;

  /** As local_eval_bv(), for boolean-sorted expressions. */
  std::optional<bool> local_eval_bool(smt_astt a) const;

  /** local_eval_bool() arms, split out for the complexity gate. */
  std::optional<bool> local_lookup_bool(const std::string &symname) const;
  std::optional<bool>
  eval_bool_fold(smt_func_kind kind, const smtlib_smt_ast *ast) const;
  std::optional<bool> eval_bool_eq(const smtlib_smt_ast *ast) const;

  /** Walk a STORE/ITE chain for the write at `index`; nullopt at a bare
   *  array symbol, which local_model cannot represent. Inert while arrays
   *  are flattened -- no SELECT or STORE reaches this backend -- and kept
   *  for the native-array mode it was written against. */
  std::optional<BigInt>
  local_eval_array_at(smt_astt array_term, const BigInt &index) const;

  /** Reads the model solver's check-sat response on first use, so a model
   *  that local_model already covers never waits on it. Returns whether a
   *  model solver is usable; acts once, safe to call repeatedly. */
  bool ensure_model_prog_ready();
  bool model_prog_response_read = false;

  std::string formula_path;
  bool solved = false;
};

#endif /* _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H */
