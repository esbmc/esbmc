/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-symex/dynamic_allocation.h>
#include <goto-symex/goto_symex.h>
#include <util/std_expr.h>

static const expr2tc *get_object(const expr2tc &expr)
{
  if(is_symbol2t(expr))
  {
    return &expr;
  }
  if(is_member2t(expr))
  {
    return get_object(to_member2t(expr).source_value);
  }
  else if(is_index2t(expr))
  {
    return get_object(to_index2t(expr).source_value);
  }

  return nullptr;
}

void goto_symext::replace_dynamic_allocation(expr2tc &expr)
{
  if(is_nil_expr(expr))
    return;

  expr->Foreach_operand([this](expr2tc &e) { replace_dynamic_allocation(e); });

  if(is_valid_object2t(expr) || is_deallocated_obj2t(expr))
  {
    expr2tc &obj_ref = (is_valid_object2t(expr))
                         ? to_valid_object2t(expr).value
                         : to_deallocated_obj2t(expr).value;

    // check what we have
    if(is_address_of2t(obj_ref))
    {
      expr2tc &obj_operand = to_address_of2t(obj_ref).ptr_obj;

      // see if that is a good one!
      const expr2tc *identifier = get_object(obj_operand);

      if(identifier != nullptr)
      {
        expr2tc base_ident = *identifier;
        cur_state->get_original_name(base_ident);

        const symbolt &symbol = ns.lookup(to_symbol2t(*identifier).thename);

        // dynamic?
        if(symbol.type.dynamic())
        {
          // TODO
        }
        else
        {
          expr = is_valid_object(symbol) ? gen_true_expr() : gen_false_expr();
          return; // done
        }
      }
    }

    // default behavior
    default_replace_dynamic_allocation(expr);
  }
  else if(is_dynamic_size2t(expr))
  {
    // default behavior
    default_replace_dynamic_allocation(expr);
  }
  else if(is_invalid_pointer2t(expr))
  {
    // default behavior
    default_replace_dynamic_allocation(expr);
  }
}

bool goto_symext::is_valid_object(const symbolt &symbol)
{
  if(symbol.static_lifetime)
    return true; // global

  // dynamic?
  if(symbol.type.dynamic())
    return false;

// current location?
#if 0
  // XXX jmorse - disabled on moving local_variables to name records. It only
  // ever contains l1 names; any lookup of symbol.name isn't going to work
  // because that's a global name.
  //
  // XXX re-enable to be able to check for stack-var-out-of-scope problems
  if(cur_state->source.is_set &&
     cur_state->source.pc->local_variables.find(symbol.name)!=
     cur_state->source.pc->local_variables.end())
    return true;

  // search call stack frames
  for(goto_symext::statet::call_stackt::const_iterator
      it=cur_state->call_stack.begin();
      it!=cur_state->call_stack.end();
      it++)
    if(it->calling_location.is_set &&
       it->calling_location.pc->local_variables.find(symbol.name)!=
       it->calling_location.pc->local_variables.end())
      return true;
#endif

  return false;
}
