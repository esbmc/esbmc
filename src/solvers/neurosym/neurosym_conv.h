#ifndef _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H
#define _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H

#include <solvers/smtlib/smtlib_conv.h>

#include <map>
#include <optional>

/** Backend for NeuroSym, a neural-guided SMT solver (a GAN proposes candidate
 *  models, with a Z3 fallback preserving soundness and completeness).
 *  NeuroSym natively parses only the QF_BV and QF_LIA fragments of SMT-LIB2,
 *  and this backend drives it as a pure QF_BV solver. NeuroSym runs as a
 *  separate program that cannot be linked into another application, so this
 *  backend reuses the smtlib backend's SMT-LIB2 serializer to render the
 *  formula into a file and runs NeuroSym on it in one-shot batch mode
 *  (--neurosym-prog, "%f" is replaced by the file path).
 *
 *  NeuroSym has no native array, floating-point, or tuple support, so the
 *  factory leaves those capability interfaces unset and ESBMC's flatteners
 *  lower everything to pure QF_BV; the header emitted before the formula is
 *  overridden to (set-logic QF_BV) accordingly.
 *  Integer/real encoding (--ir) is rejected in solve.cpp because the
 *  flattened int-mode logic would be QF_AUFLIRA, which NeuroSym cannot
 *  parse.
 *
 *  NeuroSym's own batch stdout already prints a sort-correct SMT-LIB2
 *  "(model (define-fun NAME () SORT VALUE) ...)" block on a sat verdict, so
 *  dec_solve() parses it directly into local_model instead of always paying
 *  for a second, independent solve through --neurosym-model-prog just to
 *  answer (get-value) queries. local_model only maps plain declared symbols
 *  to values though -- a query for a *composite* expression (an array
 *  select/store chain, pointer-offset arithmetic built from bit-vector ops)
 *  has no single symbol to look up. get_bv()/l_get() handle that case with
 *  local_eval(): a small recursive evaluator that walks the smtlib_smt_ast
 *  tree ESBMC handed back for the query, using local_model as the leaf
 *  values, and computes the composite result directly -- covering the
 *  common bit-vector-arithmetic and array-theory node kinds a real
 *  counterexample trace asks for. Anything local_eval() does not recognize
 *  (floating-point, uninterpreted functions/tuples) falls through to
 *  --neurosym-model-prog exactly as before it existed. */
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

  /** Look up a plain declared symbol's raw value text in local_model,
   *  interpreted as an unsigned bit pattern (sign handling happens where
   *  the caller needs it -- BigInt here is just the bits). Returns nullopt
   *  on a miss or a value form numeric_value() cannot parse (e.g. the
   *  SMT-LIB2 "(- N)" form). */
  std::optional<BigInt> local_lookup(const std::string &symname) const;

  /** Recursively evaluate a bit-vector-sorted expression using local_model
   *  for its leaf symbols. Handles literals, bit-vector arithmetic
   *  (BVADD/SUB/MUL/UDIV/SDIV/UMOD/SMOD/SHL/LSHR/ASHR/NEG/NOT/AND/OR/XOR),
   *  EXTRACT/CONCAT and ITE. Returns
   *  nullopt the moment any subterm is a leaf symbol not in local_model, or
   *  a node kind not handled (floating-point, uninterpreted functions) --
   *  the caller falls back to --neurosym-model-prog in that case, exactly
   *  as if this evaluator did not exist. Result is always masked to the
   *  expression's own bit width, as an unsigned bit pattern; callers apply
   *  sign interpretation themselves (matching get_bv()'s is_signed
   *  parameter). */
  std::optional<BigInt> local_eval_bv(smt_astt a) const;

  /** Same as local_eval_bv(), for boolean-sorted expressions: comparisons
   *  (LT/GT/LTE/GTE and their BV-prefixed signed/unsigned variants),
   *  EQ/NOTEQ (bit-vector or boolean operands), boolean connectives
   *  (AND/OR/NOT/IMPLIES/XOR), and ITE. */
  std::optional<bool> local_eval_bool(smt_astt a) const;

  /** local_eval_bool()'s SYMBOL and AND/OR/XOR arms, split out to keep it
   *  inside the repo's cyclomatic-complexity gate. */
  std::optional<bool> local_lookup_bool(const std::string &symname) const;
  std::optional<bool>
  eval_bool_fold(smt_func_kind kind, const smtlib_smt_ast *ast) const;
  std::optional<bool> eval_bool_eq(const smtlib_smt_ast *ast) const;

  /** Inert while the factory leaves array_iface unset: ESBMC flattens every
   *  array to bit-vectors, so no SELECT or STORE node reaches this backend
   *  and neither this nor local_eval_bv()'s SELECT arm is entered. Both are
   *  kept for the native-array mode they were written against, and become
   *  live the moment that is enabled.
   *
   *  Evaluate an array-sorted term at one concrete index: walks a
   *  STORE-chain looking for a write at `index`, recursing into the base
   *  array on a miss; an ITE picks a branch by its (evaluated) condition
   *  and recurses into it with the same index. A bare array SYMBOL leaf has
   *  no representation in local_model (NeuroSym's model output only
   *  contains scalar define-funs) and returns nullopt -- an array whose
   *  full contents were never constrained by a store the trace actually
   *  reads through is a real gap, not a bug, and degrades to the
   *  --neurosym-model-prog fallback like any other unhandled case. */
  std::optional<BigInt>
  local_eval_array_at(smt_astt array_term, const BigInt &index) const;

  /** Lazily reads the model solver's initial check-sat response (the
   *  handshake (get-value) queries need) the *first* time a variable is not
   *  covered by local_model -- not unconditionally in dec_solve(). If every
   *  queried variable is in local_model (the common case: NeuroSym's model
   *  output normally covers every free variable in the formula), this is
   *  never called at all, and the model solver's answer -- on a large
   *  formula, potentially itself a multi-minute solve -- is never waited
   *  for. Safe to call repeatedly; only acts once. */
  bool ensure_model_prog_ready();
  bool model_prog_response_read = false;

  std::string formula_path;
  bool solved = false;
};

#endif /* _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H */
