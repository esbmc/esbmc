#include <cassert>
#include <goto-programs/goto_inline.h>
#include <goto-programs/remove_no_op.h>
#include <langapi/language_util.h>
#include <util/base_type.h>
#include <util/cprover_prefix.h>
#include <util/prefix.h>

/// Returns true if the formal parameter type and the actual argument type are
/// compatible enough that inserting a typecast is safe (pointer-to-pointer,
/// array-to-pointer with matching subtype, or numeric/bool to numeric/bool).
static bool can_typecast_argument(const type2tc &formal, const type2tc &actual)
{
  if (is_pointer_type(formal) && is_pointer_type(actual))
    return true;

  if (is_array_type(formal) && is_pointer_type(actual))
    return to_array_type(formal).subtype == to_pointer_type(actual).subtype;

  const bool formal_numeric = is_signedbv_type(formal) ||
                              is_unsignedbv_type(formal) ||
                              is_bool_type(formal);
  const bool actual_numeric = is_signedbv_type(actual) ||
                              is_unsignedbv_type(actual) ||
                              is_bool_type(actual);
  return formal_numeric && actual_numeric;
}

void goto_inlinet::parameter_assignments(
  const locationt &location,
  const code_typet &code_type,
  const std::vector<expr2tc> &arguments,
  goto_programt &dest)
{
  const code_typet::argumentst &argument_types = code_type.arguments();

  auto actual_it = arguments.begin();
  for (const auto &argument_type : argument_types)
  {
    // The "argument_type" entry from a code_typet is itself an exprt that
    // carries the formal parameter's name (#identifier) and type.
    const exprt &formal = static_cast<const exprt &>(argument_type);
    const irep_idt &identifier = formal.cmt_identifier();
    const type2tc formal_type = migrate_type(ns.follow(formal.type()));

    // If the call site supplied fewer arguments than the function definition
    // declares, only declare the formal parameter (it remains unassigned and
    // any read of it will be treated as nondeterministic by symex).  This
    // mirrors the symex-time handling in goto_symext::argument_assignments.
    if (actual_it == arguments.end())
    {
      log_warning(
        "function call: missing argument for parameter '{}'; "
        "modelled as nondet",
        id2string(identifier));

      if (identifier != "")
      {
        goto_programt::targett decl = dest.add_instruction();
        decl->make_other();
        decl->code = code_decl2tc(formal_type, identifier);
        decl->location = location;
        decl->function = location.get_function();
      }
      continue;
    }

    // Don't assign arguments if they have no name, see regression spec21
    if (identifier == "")
    {
      ++actual_it;
      continue;
    }

    {
      goto_programt::targett decl = dest.add_instruction();
      decl->make_other();
      decl->code = code_decl2tc(formal_type, identifier);
      decl->location = location;
      decl->function = location.get_function();
    }

    // nil means "don't assign"
    if (!is_nil_expr(*actual_it))
    {
      expr2tc actual = *actual_it;

      // it should be the same exact type
      if (!base_type_eq(formal_type, actual->type, ns))
      {
        if (can_typecast_argument(formal_type, actual->type))
        {
          actual = typecast2tc(formal_type, actual);
        }
        else
        {
          log_error(
            "function call: argument `{}' type mismatch: got {}, expected {}",
            id2string(identifier),
            from_type(ns, identifier, actual->type),
            from_type(ns, identifier, formal_type));
          abort();
        }
      }

      goto_programt::targett assignment = dest.add_instruction(ASSIGN);
      assignment->location = location;
      assignment->code =
        code_assign2tc(symbol2tc(formal_type, identifier), actual);
      assignment->function = location.get_function();
    }

    ++actual_it;
  }

  // too many arguments -- we just ignore that, no harm done
}

void goto_inlinet::replace_return(goto_programt &dest, const expr2tc &lhs)
{
  for (goto_programt::instructionst::iterator it = dest.instructions.begin();
       it != dest.instructions.end();
       ++it)
  {
    if (!it->is_return())
      continue;

    const code_return2t &ret = to_code_return2t(it->code);

    if (!is_nil_expr(lhs))
    {
      expr2tc rhs = ret.operand;

      // this may happen if the declared return type at the call site
      // differs from the defined return type
      if (lhs->type != rhs->type)
        rhs = typecast2tc(lhs->type, rhs);

      goto_programt tmp;
      goto_programt::targett assignment = tmp.add_instruction(ASSIGN);
      assignment->code = code_assign2tc(lhs, rhs);
      assignment->location = it->location;
      assignment->function = it->location.get_function();

      dest.insert_swap(it, *assignment);
      ++it;
    }
    else if (!is_nil_expr(ret.operand))
    {
      // Encode evaluation of return expr, so that returns with pointer
      // derefs in them still get dereferenced, even when the result is
      // discarded.
      goto_programt tmp;
      goto_programt::targett expression = tmp.add_instruction(OTHER);
      expression->make_other();
      expression->location = it->location;
      expression->function = it->location.get_function();
      expression->code = code_expression2tc(ret.operand);

      dest.insert_swap(it, *expression);
      ++it;
    }

    it->make_goto(--dest.instructions.end());
  }
}

void goto_inlinet::expand_function_call(
  goto_programt &dest,
  goto_programt::targett &target,
  const expr2tc &lhs,
  const expr2tc &function,
  const std::vector<expr2tc> &arguments,
  bool full)
{
  if (!is_symbol2t(function))
  {
    log_error(
      "function_call expects symbol as function operand, but got `{}'",
      get_expr_id(function));
    abort();
  }

  const irep_idt &identifier = to_symbol2t(function).thename;

  // see if we are already expanding it
  if (recursion_set.find(identifier) != recursion_set.end())
  {
    if (!full)
    {
      ++target;
      return; // simply ignore, we don't do full inlining, it's ok
    }

    // it's really recursive. Give up.
    log_warning("Recursion is ignored when inlining");
    target->make_skip();
    ++target;
    return;
  }

  goto_functionst::function_mapt::iterator m_it =
    goto_functions.function_map.find(identifier);

  if (m_it == goto_functions.function_map.end())
  {
    log_error("failed to find function `{}'", id2string(identifier));
    abort();
  }

  goto_functiont &f = m_it->second;

  // see if we need to inline this
  if (!full)
  {
    if (!f.body_available || (f.body.instructions.size() > smallfunc_limit))
    {
      ++target;
      return;
    }
  }

  if (f.body_available)
  {
    inlined_funcs.insert(identifier.as_string());
    for (const auto &inlined_func : f.inlined_funcs)
      inlined_funcs.insert(inlined_func);

    recursion_sett::iterator recursion_it =
      recursion_set.insert(identifier).first;

    goto_programt tmp2;
    tmp2.copy_from(f.body);

    assert(tmp2.instructions.back().is_end_function());
    tmp2.instructions.back().type = LOCATION;

    replace_return(tmp2, lhs);

    goto_programt tmp;
    parameter_assignments(
      tmp2.instructions.front().location, f.type, arguments, tmp);
    tmp.destructive_append(tmp2);

    if (f.body.hide)
    {
      const locationt &call_site = target->location;

      Forall_goto_program_instructions (it, tmp)
      {
        if (call_site.is_not_nil())
        {
          // can't just copy, e.g., due to comments field
          it->location.id(""); // not NIL
          it->location.set_file(call_site.get_file());
          it->location.set_line(call_site.get_line());
          it->location.set_column(call_site.get_column());
          it->location.set_function(call_site.get_function());
        }
      }
    }

    // do this recursively
    goto_inline_rec(tmp, full);

    // set up location instruction for function call
    target->type = LOCATION;
    target->code = expr2tc();

    goto_programt::targett next_target(target);
    next_target++;
    dest.instructions.splice(next_target, tmp.instructions);
    target = next_target;

    recursion_set.erase(recursion_it);
  }
  else
  {
    if (no_body_set.insert(identifier).second)
      log_warning("no body for function `{}'", id2string(identifier));

    goto_programt tmp;

    // evaluate function arguments -- they might have
    // pointer dereferencing or the like
    for (const auto &arg : arguments)
    {
      goto_programt::targett t = tmp.add_instruction();
      t->make_other();
      t->location = target->location;
      t->function = target->location.get_function();
      t->code = code_expression2tc(arg);
    }

    // return value
    if (!is_nil_expr(lhs))
    {
      goto_programt::targett t = tmp.add_instruction(ASSIGN);
      t->location = target->location;
      t->function = target->location.get_function();
      t->code = code_assign2tc(lhs, gen_nondet(lhs->type));
    }

    // now just kill call
    target->type = LOCATION;
    target->code = expr2tc();
    ++target;

    dest.instructions.splice(target, tmp.instructions);
  }
}

void goto_inlinet::goto_inline(goto_programt &dest)
{
  goto_inline_rec(dest, true);
  replace_return(dest, expr2tc());
}

void goto_inlinet::goto_inline_rec(goto_programt &dest, bool full)
{
  bool changed = false;

  for (goto_programt::instructionst::iterator it = dest.instructions.begin();
       it != dest.instructions.end();)
  {
    if (inline_instruction(dest, full, it))
      changed = true;
    else
      ++it;
  }

  if (changed)
  {
    remove_no_op(dest);
    dest.update();
  }
}

bool goto_inlinet::inline_instruction(
  goto_programt &dest,
  bool full,
  goto_programt::targett &it)
{
  if (it->is_function_call())
  {
    const code_function_call2t &call = to_code_function_call2t(it->code);

    if (is_symbol2t(call.function))
    {
      expand_function_call(
        dest, it, call.ret, call.function, call.operands, full);
      return true;
    }
  }

  return false;
}

void goto_inline(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns,
  goto_programt &dest)
{
  goto_inlinet goto_inline(goto_functions, options, ns);

  {
    // find main
    goto_functionst::function_mapt::const_iterator it =
      goto_functions.function_map.find("__ESBMC_main");

    if (it == goto_functions.function_map.end())
    {
      dest.clear();
      return;
    }

    dest.copy_from(it->second.body);
  }
  goto_inline.goto_inline(dest);

  // clean up
  for (auto &it : goto_functions.function_map)
    if (it.first != "__ESBMC_main")
    {
      it.second.body_available = false;
      it.second.body.clear();
    }
}

void goto_inline(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns)
{
  goto_inlinet goto_inline(goto_functions, options, ns);

  // find main
  goto_functionst::function_mapt::iterator it =
    goto_functions.function_map.find("__ESBMC_main");

  if (it == goto_functions.function_map.end())
    return;

  goto_inline.goto_inline(it->second.body);

  // clean up
  for (auto &it : goto_functions.function_map)
    if (it.first != "main")
    {
      it.second.body_available = false;
      it.second.body.clear();
    }
}

void goto_partial_inline(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns,
  unsigned _smallfunc_limit)
{
  goto_inlinet goto_inline(goto_functions, options, ns);

  goto_inline.smallfunc_limit = _smallfunc_limit;

  for (auto &it : goto_functions.function_map)
  {
    goto_inline.inlined_funcs.clear();
    if (it.second.body_available)
      goto_inline.goto_inline_rec(it.second.body, false);
    it.second.inlined_funcs = goto_inline.inlined_funcs;
  }
}
