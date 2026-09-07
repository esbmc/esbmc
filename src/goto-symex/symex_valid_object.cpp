#include <goto-symex/dynamic_allocation.h>
#include <goto-symex/goto_symex.h>
#include <util/irep/std_expr.h>

static const expr2tc *get_object(const expr2tc &expr)
{
  if (is_symbol2t(expr))
  {
    return &expr;
  }
  if (is_member2t(expr))
  {
    return get_object(to_member2t(expr).source_value);
  }
  else if (is_index2t(expr))
  {
    return get_object(to_index2t(expr).source_value);
  }

  return nullptr;
}

/// The named object \p ptr addresses, resolved through the value set, or nil.
///
/// A pointer reaching a check through a parameter or a local variable is a
/// plain symbol, so the syntactic address_of tests below find no name and the
/// check falls back on __ESBMC_alloc and DYNAMIC_SIZE. Neither is maintained
/// for automatic storage, so the solver may pick a live stack object invalid
/// or size it arbitrarily, and a caller cannot discharge an is_fresh
/// precondition it plainly satisfies (#7464).
expr2tc goto_symext::value_set_named_object(const expr2tc &ptr)
{
  expr2tc p = ptr;
  while (is_typecast2t(p))
    p = to_typecast2t(p).from;

  if (is_address_of2t(p) || !is_pointer_type(p->type))
    return expr2tc();

  value_setst::valuest targets;
  cur_state->value_set.get_value_set(p, targets);
  if (targets.size() != 1 || !is_object_descriptor2t(targets.front()))
    return expr2tc();

  const expr2tc &obj = to_object_descriptor2t(targets.front()).object;
  if (is_nil_expr(obj))
    return expr2tc();

  const expr2tc *base = get_object(obj);
  if (base == nullptr)
    return expr2tc();

  const symbolt *sym = ns.lookup(to_symbol2t(*base).thename);
  if (sym == nullptr || sym->get_type().dynamic())
    return expr2tc();

  // The enforce harness backs a pointer parameter whose contract states no
  // extent with one element of stack (emit_struct_stack_backing, #6212). That
  // object stands in for whatever the caller would pass; reading an extent off
  // it would hand the contract the very guarantee it failed to state.
  if (id2string(sym->id).rfind("__ESBMC_harness_ptr_", 0) == 0)
    return expr2tc();

  return *base;
}

/// Decide VALID_OBJECT from the object the value set names, if it names one.
///
/// is_valid_object() answers false for every automatic object: the scope
/// tracking it would need is #if 0'd out at the end of this file. The syntactic
/// path never reaches it, because a named object returns its declared extent
/// and drops this conjunct. Do the same once the value set has named the
/// object: a pointer it resolves addresses a live object, and the extent is
/// checked separately (#7464).
bool goto_symext::resolve_valid_object_by_value_set(
  expr2tc &expr,
  const expr2tc &obj_ref)
{
  if (is_nil_expr(value_set_named_object(obj_ref)))
    return false;

  expr = gen_true_expr();
  return true;
}

/// Size an object with automatic or static storage by its type.
///
/// DYNAMIC_SIZE is maintained for the heap alone, so it says nothing about the
/// extent an is_fresh precondition has to compare against for such an object
/// (#7464).
bool goto_symext::resolve_dynamic_size_by_value_set(expr2tc &expr)
{
  expr2tc named = value_set_named_object(to_dynamic_size2t(expr).value);
  if (is_nil_expr(named))
    return false;

  expr = constant_int2tc(size_type2(), type_byte_size(named->type, &ns));
  return true;
}

/// Decide VALID_OBJECT / DEALLOCATED_OBJ against the object the pointer names.
void goto_symext::replace_valid_object(expr2tc &expr)
{
  expr2tc &obj_ref = (is_valid_object2t(expr))
                       ? to_valid_object2t(expr).value
                       : to_deallocated_obj2t(expr).value;

  if (resolve_valid_object_by_value_set(expr, obj_ref))
    return;

  if (is_address_of2t(obj_ref))
  {
    expr2tc &obj_operand = to_address_of2t(obj_ref).ptr_obj;

    const expr2tc *identifier = get_object(obj_operand);

    if (identifier != nullptr)
    {
      expr2tc base_ident = *identifier;
      cur_state->get_original_name(base_ident);

      const symbolt &symbol = *ns.lookup(to_symbol2t(*identifier).thename);

      // dynamic?
      if (symbol.get_type().dynamic())
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

/// Size the object DYNAMIC_SIZE names, falling back to the heap-only table.
void goto_symext::replace_dynamic_size(expr2tc &expr)
{
  if (!resolve_dynamic_size_by_value_set(expr))
    default_replace_dynamic_allocation(expr);
}

void goto_symext::replace_dynamic_allocation(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  expr->Foreach_operand([this](expr2tc &e) { replace_dynamic_allocation(e); });

  if (is_valid_object2t(expr) || is_deallocated_obj2t(expr))
  {
    replace_valid_object(expr);
  }
  else if (is_dynamic_size2t(expr))
  {
    replace_dynamic_size(expr);
  }
  else if (is_invalid_pointer2t(expr))
  {
    // default behavior
    default_replace_dynamic_allocation(expr);
  }
  else if (is_capability_base2t(expr) || is_capability_top2t(expr))
  {
    default_replace_dynamic_allocation(expr);
  }
  else if (is_ptr_mem2t(expr))
  {
    default_replace_dynamic_allocation(expr);
  }
}

bool goto_symext::is_valid_object(const symbolt &symbol)
{
  if (symbol.static_lifetime)
    return true; // global

  // dynamic?
  if (symbol.get_type().dynamic())
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
