#ifndef _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H
#define _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H

#include <solvers/smtlib/smtlib_conv.h>

#include <map>

/** Backend for NeuroSym, a neural-guided SMT solver (a GAN proposes candidate
 *  models, with a Z3 fallback preserving soundness and completeness). NeuroSym
 *  natively parses the QF_BV, QF_ABV, and QF_LIA fragments of SMT-LIB2
 *  (arrays via a read-over-write bit-blaster encoding, not a full decision
 *  procedure), and this backend drives it as a QF_ABV solver. NeuroSym is a
 *  Python program that cannot be linked into another application, so this
 *  backend reuses the smtlib backend's SMT-LIB2 serializer to render the
 *  formula into a file and runs NeuroSym on it in one-shot batch mode
 *  (--neurosym-prog, "%f" is replaced by the file path).
 *
 *  Arrays are enabled (array_api set in the factory): neurosym_convt
 *  inherits array_iface from smtlib_convt, which already serializes native
 *  (Array ...) / select / store syntax, so ESBMC passes arrays through
 *  instead of pre-flattening every access into per-index bit-vector
 *  equality/implication chains -- for an array-heavy program that
 *  flattening can otherwise blow a modest formula up into a CNF with well
 *  over a million boolean variables. NeuroSym has no native floating-point
 *  or tuple support, so those two interfaces stay unset and ESBMC's
 *  flatteners still lower them to pure bit-vectors; the header emitted
 *  before the formula is overridden to (set-logic QF_ABV) accordingly.
 *  Integer/real encoding (--ir) is rejected in solve.cpp because the
 *  flattened int-mode logic would be QF_AUFLIRA, which NeuroSym cannot
 *  parse.
 *
 *  NeuroSym's own batch stdout already prints a sort-correct SMT-LIB2
 *  "(model (define-fun NAME () SORT VALUE) ...)" block on a sat verdict, so
 *  dec_solve() parses it directly into local_model instead of always paying
 *  for a second, independent solve through --neurosym-model-prog just to
 *  answer (get-value) queries -- on a real captured formula (24M CNF
 *  variables) that second solve, through a plain interactive SMT-LIB2 pipe,
 *  measured slower than NeuroSym's own batch solve of the same formula.
 *  --neurosym-model-prog remains supported as a fallback for any variable
 *  local parsing did not cover (e.g. an unrecognized sort). Without either a
 *  usable local model or a model solver, satisfiable results require
 *  --result-only. */
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

  /** Parse a "(model (define-fun NAME () SORT VALUE) ...)" block -- the
   *  format NeuroSym's own format_output() emits (main.py / ns_solver.py) --
   *  into local_model. Tolerant of the whole block being absent or of
   *  individual entries it cannot make sense of: those variables simply
   *  fall through to the --neurosym-model-prog fallback, the same as if
   *  local parsing were not attempted at all. */
  void parse_model_block(const std::string &output);

  /** local_model[name] holds the *raw* value text as NeuroSym printed it --
   *  a decimal numeral (Int) or a sized hex literal (BitVec), matching what
   *  smtlib_convt::get_bv()/l_get() already know how to interpret from a
   *  real solver's (get-value) response, via the same interp_numeric()
   *  helper. Bool is stored as "true"/"false". */
  std::map<std::string, std::string> local_model;

  /** Lazily reads the model solver's initial check-sat response (the
   *  handshake (get-value) queries need) the *first* time a variable is not
   *  covered by local_model -- not unconditionally in dec_solve(). If every
   *  queried variable is in local_model (the common case: NeuroSym's model
   *  output normally covers every free variable in the formula), this is
   *  never called at all, and the model solver's answer -- on a large
   *  formula, potentially itself a multi-minute solve -- is never waited
   *  for. Safe to call repeatedly; only acts once. */
  void ensure_model_prog_ready();
  bool model_prog_response_read = false;

  std::string formula_path;
  bool solved = false;
};

#endif /* _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H */
