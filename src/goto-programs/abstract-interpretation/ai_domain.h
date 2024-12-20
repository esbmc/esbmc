/// \file
/// Abstract Interpretation Domain

#ifndef CPROVER_ANALYSES_AI_DOMAIN_H
#define CPROVER_ANALYSES_AI_DOMAIN_H

#include <goto-programs/goto_program.h>
#include <irep2/irep2_utils.h>

// forward reference the abstract interpreter interface
class ai_baset;

/// The interface offered by a domain, allows code to manipulate domains without
/// knowing their exact type.  Derive from this to implement domains.
class ai_domain_baset
{
protected:
  /// The constructor is expected to produce 'false' or 'bottom'
  ai_domain_baset()
  {
  }

public:
  virtual ~ai_domain_baset()
  {
  }

  /// how function calls are treated:
  /// a) there is an edge from each call site to the function head
  /// b) there is an edge from the last instruction (END_FUNCTION)
  ///    of the function to the instruction _following_ the call site
  ///    (this also needs to set the LHS, if applicable)
  ///
  /// "this" is the domain before the instruction "from"
  /// "from" is the instruction to be interpreted
  /// "to" is the next instruction (for GOTO, FUNCTION_CALL, END_FUNCTION)
  ///
  /// PRECONDITION(from.is_dereferenceable(), "Must not be _::end()")
  /// PRECONDITION(to.is_dereferenceable(), "Must not be _::end()")
  /// PRECONDITION(are_comparable(from,to) ||
  ///              (from->is_function_call() || from->is_end_function())

  virtual void transform(
    goto_programt::const_targett from,
    goto_programt::const_targett to,
    ai_baset &ai,
    const namespacet &ns) = 0;

  virtual void output(std::ostream &out) const = 0;

  /// no states
  virtual void make_bottom() = 0;

  /// all states -- the analysis doesn't use this,
  /// and domains may refuse to implement it.
  virtual void make_top() = 0;

  /// a reasonable entry-point state
  virtual void make_entry() = 0;

  virtual bool is_bottom() const = 0;

  virtual bool is_top() const = 0;

  /// also add
  ///
  ///   bool merge(const T &b, const_targett from, const_targett to);
  ///
  /// This computes the join between "this" and "b".
  /// Return true if "this" has changed.
  /// In the usual case, "b" is the updated state after "from"
  /// and "this" is the state before "to".
  ///
  /// PRECONDITION(from.is_dereferenceable(), "Must not be _::end()")
  /// PRECONDITION(to.is_dereferenceable(), "Must not be _::end()")

  /// This method allows an expression to be simplified / evaluated using the
  /// current state.  It is used to evaluate assertions and in program
  /// simplification

  /// return true if unchanged
  virtual bool ai_simplify(expr2tc &condition, const namespacet &ns) const = 0;

  /// Simplifies the expression but keeps it as an l-value
  virtual bool ai_simplify_lhs(expr2tc &condition, const namespacet &ns) const;

  /// Gives a Boolean condition that is true for all values represented by the
  /// domain.  This allows domains to be converted into program invariants.
  virtual expr2tc to_predicate(void) const
  {
    if (is_bottom())
      return gen_false_expr();
    return gen_true_expr();
  }
};

#endif
