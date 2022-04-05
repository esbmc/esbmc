#include <goto-symex/slice.h>

namespace
{
/**
 * @brief Class for the symex-slicer, this slicer is to be executed
 * on SSA formula in order to remove every symbol that does not depends
 * on it
 *
 * It works by constructing a symbol dependency list by transversing
 * the SSA formula in reverse order. If any assume, assignment, or renumber
 * step does not belong into this dependency, then it will be ignored.
 */
class symex_slicet
{
public:
  explicit symex_slicet(bool assume);

  /**
   * Iterate over all steps of the \eq in REVERSE order,
   * getting symbol dependencies. If an
   * assignment, renumber or assume does not contain one
   * of the dependency symbols, then it will be ignored.
   *
   * @param eq symex formula to be sliced
   */
  void slice(std::shared_ptr<symex_target_equationt> &eq);

  /**
   * Holds the symbols the current equation depends on.
   */
  std::unordered_set<std::string> depends;

  BigInt ignored; /// tracks how many steps were sliced

protected:
  bool slice_assumes;

  /**
   * Recursively explores the operands of an expression \expr
   * If a symbol is found, then it is added into the #depends
   * member if `Add` is true, otherwise returns true.
   *
   * @param expr expression to extract every symbol
   * @return true if at least one symbol was found
   */
  template <bool Add>
  bool get_symbols(const expr2tc &expr);

  /**
   * Helper function, it is used to select specialization will
   * be used, i.e. assume, assignment or renumber
   *
   * Note 1: ASSERTS are not sliced, only their symbols are added
   * into the #depends
   *
   * Note 2: Similar to ASSERTS, if 'slice-assumes' option is
   * is not enabled. Then only its symbols are added into the
   * #depends
   *
   * TODO: All slice specialization can be converted into a lambda
   *
   * @param SSA_step any kind of SSA expression
   */
  void slice(symex_target_equationt::SSA_stept &SSA_step);

  /**
   * Remove unneeded assumes from the formula
   *
   * Check if the Assume cond symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * Note 1: All the conditions operands are going to be added
   * into the #depends. This makes that the condition itself as
   * a "reverse taint"
   *
   * TODO: What happens if the ASSUME would result in false?
   *
   * @param SSA_step an assume step
   */
  void slice_assume(symex_target_equationt::SSA_stept &SSA_step);

  /**
   * Remove unneeded assignments from the formula
   *
   * Check if the LHS symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * @param SSA_step an assignment step
   */
  void slice_assignment(symex_target_equationt::SSA_stept &SSA_step);

  /**
   * Remove unneeded renumbers from the formula
   *
   * Check if the LHS symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * @param SSA_step an renumber step
   */
  void slice_renumber(symex_target_equationt::SSA_stept &SSA_step);
};

} /* end anonymous namespace */

symex_slicet::symex_slicet(bool assume) : ignored(0), slice_assumes(assume)
{
}

static bool no_slice(const symbol2t &sym)
{
  return config.no_slice_names.count(sym.thename.as_string()) ||
         config.no_slice_ids.count(sym.get_symbol_name());
}

template <bool Add>
bool symex_slicet::get_symbols(const expr2tc &expr)
{
  bool res = false;
  // Recursively look if any of the operands has a inner symbol
  expr->foreach_operand([this, &res](const expr2tc &e) {
    if(!is_nil_expr(e))
      res |= get_symbols<Add>(e);
    return res;
  });

  if(!is_symbol2t(expr))
    return res;

  const symbol2t &s = to_symbol2t(expr);
  if constexpr(Add)
    res |= depends.insert(s.get_symbol_name()).second;
  else
    res |= no_slice(s) || depends.find(s.get_symbol_name()) != depends.end();
  return res;
}

void symex_slicet::slice(std::shared_ptr<symex_target_equationt> &eq)
{
  depends.clear();

  for(symex_target_equationt::SSA_stepst::reverse_iterator it =
        eq->SSA_steps.rbegin();
      it != eq->SSA_steps.rend();
      it++)
    slice(*it);
}

void symex_slicet::slice(symex_target_equationt::SSA_stept &SSA_step)
{
  switch(SSA_step.type)
  {
  case goto_trace_stept::ASSERT:
    get_symbols<true>(SSA_step.guard);
    get_symbols<true>(SSA_step.cond);
    break;

  case goto_trace_stept::ASSUME:
    if(slice_assumes)
      slice_assume(SSA_step);
    else
    {
      get_symbols<true>(SSA_step.guard);
      get_symbols<true>(SSA_step.cond);
    }
    break;

  case goto_trace_stept::ASSIGNMENT:
    slice_assignment(SSA_step);
    break;

  case goto_trace_stept::OUTPUT:
    break;

  case goto_trace_stept::RENUMBER:
    slice_renumber(SSA_step);
    break;

  default:
    assert(false);
  }
}

void symex_slicet::slice_assume(symex_target_equationt::SSA_stept &SSA_step)
{
  if(!get_symbols<false>(SSA_step.cond))
  {
    // we don't really need it
    SSA_step.ignore = true;
    ++ignored;
    if(is_symbol2t(SSA_step.cond))
      log_debug(
        "slice ignoring assume symbol {}",
        to_symbol2t(SSA_step.cond).get_symbol_name());
    else
      log_debug("slide ignoring assume expression");
  }
  else
  {
    // If we need it, add the symbols to dependency
    get_symbols<true>(SSA_step.guard);
    get_symbols<true>(SSA_step.cond);
  }
}

void symex_slicet::slice_assignment(symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));
  // TODO: create an option to ignore nondet symbols (test case generation)

  if(!get_symbols<false>(SSA_step.lhs))
  {
    // we don't really need it
    SSA_step.ignore = true;
    ++ignored;
    log_debug(
      "slice ignoring assignment to symbol {}",
      to_symbol2t(SSA_step.lhs).get_symbol_name());
  }
  else
  {
    get_symbols<true>(SSA_step.guard);
    get_symbols<true>(SSA_step.rhs);

    // Remove this symbol as we won't be seeing any references to it further
    // into the history.
    depends.erase(to_symbol2t(SSA_step.lhs).get_symbol_name());
  }
}

void symex_slicet::slice_renumber(symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));

  if(!get_symbols<false>(SSA_step.lhs))
  {
    // we don't really need it
    SSA_step.ignore = true;
    ++ignored;
    log_debug(
      "slice ignoring renumbering symbol {}",
      to_symbol2t(SSA_step.lhs).get_symbol_name());
  }

  // Don't collect the symbol; this insn has no effect on dependencies.
}

/**
 * Naive slicer: slice every step after the last assertion
 * @param eq symex formula to be sliced
 * @return number of steps that were ignored
 */
static BigInt simple_slice(std::shared_ptr<symex_target_equationt> &eq)
{
  BigInt ignored = 0;

  // just find the last assertion
  symex_target_equationt::SSA_stepst::iterator last_assertion =
    eq->SSA_steps.end();

  for(symex_target_equationt::SSA_stepst::iterator it = eq->SSA_steps.begin();
      it != eq->SSA_steps.end();
      it++)
    if(it->is_assert())
      last_assertion = it;

  // slice away anything after it

  symex_target_equationt::SSA_stepst::iterator s_it = last_assertion;

  if(s_it != eq->SSA_steps.end())
    for(s_it++; s_it != eq->SSA_steps.end(); s_it++)
    {
      s_it->ignore = true;
      ++ignored;
    }

  return ignored;
}

BigInt slice(std::shared_ptr<symex_target_equationt> &eq)
{
  const optionst &opts = config.options;

  if(opts.get_bool_option("no-slice"))
    return simple_slice(eq);

  symex_slicet symex_slice(opts.get_bool_option("slice-assumes"));
  symex_slice.slice(eq);
  return symex_slice.ignored;
}
