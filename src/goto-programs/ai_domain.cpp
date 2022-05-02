/*******************************************************************\

Module: Abstract Interpretation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

/// \file
/// Abstract Interpretation Domain

#include <goto-programs/ai_domain.h>
#include <util/simplify_expr.h>

/// Use the information in the domain to simplify the expression on the LHS of
/// an assignment. This for example won't simplify symbols to their values, but
/// does simplify indices in arrays, members of structs and dereferencing of
/// pointers
/// \param condition: the expression to simplify
/// \param ns: the namespace
/// \return True if condition did not change. False otherwise. condition will be
///   updated with the simplified condition if it has worked
bool ai_domain_baset::ai_simplify_lhs(expr2tc &condition, const namespacet &ns)
  const
{
  // Care must be taken here to give something that is still writable
  if(is_index2t(condition))
  {
    bool no_simplification = ai_simplify(to_index2t(condition).index, ns);
    if(!no_simplification)
      simplify(condition);

    return no_simplification;
  }

  if(is_dereference2t(condition))
  {
    bool no_simplification = ai_simplify(to_dereference2t(condition).value, ns);
    if(!no_simplification)
      simplify(condition);

    return no_simplification;
  }

  if(is_member2t(condition))
  {
    // Since simplify_ai_lhs is required to return an addressable object
    // (so remains a valid left hand side), to simplify
    // `(something_simplifiable).b` we require that `something_simplifiable`
    // must also be addressable
    bool no_simplification =
      ai_simplify_lhs(to_member2t(condition).source_value, ns);
    if(!no_simplification)
      simplify(condition);

    return no_simplification;
  }

  return true;
}
