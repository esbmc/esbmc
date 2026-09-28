#ifndef CPROVER_POINTER_ANALYSIS_VALUE_SETS_H
#define CPROVER_POINTER_ANALYSIS_VALUE_SETS_H

#include <goto-programs/goto_program.h>
#include <set>
#include <irep2/irep2.h>

// an abstract base class

class value_setst
{
public:
  value_setst() = default;

  typedef std::list<expr2tc> valuest;
  typedef goto_programt::const_targett locationt;

  // this is not const to allow a lazy evaluation
  virtual void get_values(locationt l, const expr2tc &expr, valuest &dest) = 0;

  /// The objects a dereference-style l-value may refer to, as
  /// `object_descriptor2t`s (or a raw unknown/invalid expression when no
  /// nameable object could be determined).
  virtual void
  get_reference_set(locationt l, const expr2tc &expr, valuest &dest) = 0;

  /// True iff \p l carries points-to information; consumers must abstain when
  /// it does not.
  virtual bool has_location(locationt l) const = 0;

  virtual ~value_setst() = default;
};

#endif
