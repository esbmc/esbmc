#include <goto-symex/goto_symex.h>
#include <irep2/irep2.h>
#include <util/migrate.h>

void goto_symext::symex_catch()
{
  // there are two variants: 'push' and 'pop'
  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  if (instruction.targets.empty()) // The second catch, pop from the stack
  {
    // Pop from the stack
    cur_state->stack_catch.pop();

    // Increase the program counter
    cur_state->source.pc++;
  }
  else // The first catch, push it to the stack
  {
    goto_symex_statet::exceptiont exception;

    // copy targets
    const code_cpp_catch2t &catch_ref = to_code_cpp_catch2t(instruction.code);

    assert(catch_ref.exception_list.size() == instruction.targets.size());

    // Fill the map with the catch type and the target
    unsigned i = 0;
    for (goto_programt::targetst::const_iterator it =
           instruction.targets.begin();
         it != instruction.targets.end();
         ++it, ++i)
    {
      exception.catch_map[catch_ref.exception_list[i]] = *it;
      exception.catch_order[catch_ref.exception_list[i]] = i;
    }

    // Remember which function frame owns this try/catch region, so exception
    // dispatch can tell which frames an exception exits before being caught
    // here and enforce their specifications at the boundary.
    assert(!cur_state->call_stack.empty());
    exception.owner_frame_depth = cur_state->call_stack.size() - 1;

    // Stack it
    cur_state->stack_catch.push(exception);

    // Increase the program counter
    cur_state->source.pc++;
  }
}

bool goto_symext::is_python_exception_subtype(
  const irep_idt &thrown_type,
  const irep_idt &catch_type)
{
  std::string thrown = thrown_type.as_string();
  std::string catch_t = catch_type.as_string();

  // Exact match
  if (thrown == catch_t)
    return true;

  // Look up the thrown exception class in the symbol table
  const symbolt *thrown_symbol = ns.lookup("tag-" + thrown);
  if (!thrown_symbol)
    return false;

  // Get the bases from the type metadata
  const irept &bases = thrown_symbol->get_type().find("bases");

  if (bases.is_nil())
    return false;

  const irept::subt &base_list = bases.get_sub();

  // Iterate through base classes
  for (const auto &base : base_list)
  {
    // The base class is stored with id() == "tag-BaseClassName"
    std::string base_name = base.id().as_string();

    // Remove "tag-" prefix to get the class name
    if (base_name.starts_with("tag-"))
      base_name = base_name.substr(4);

    // Recursively check if this base class matches or inherits from catch_type
    if (is_python_exception_subtype(irep_idt(base_name), catch_type))
      return true;
  }

  return false;
}

bool goto_symext::exception_caught_in_top(
  const std::vector<irep_idt> &exception_list,
  std::size_t &owner_frame_depth)
{
  if (cur_state->stack_catch.empty())
    return false;

  const goto_symex_statet::exceptiont &top = cur_state->stack_catch.top();

  for (const auto &it : exception_list)
  {
    // Exact match, or (for Python exceptions) a base-class match.
    if (top.catch_map.find(it) != top.catch_map.end())
    {
      owner_frame_depth = top.owner_frame_depth;
      return true;
    }
    for (const auto &entry : top.catch_map)
      if (is_python_exception_subtype(it, entry.first))
      {
        owner_frame_depth = top.owner_frame_depth;
        return true;
      }

    // A pointer can be caught by catch(void*); anything by catch(...).
    if (
      (it.as_string().find("_ptr") != std::string::npos &&
       top.catch_map.find("void_ptr") != top.catch_map.end()) ||
      top.catch_map.find("ellipsis") != top.catch_map.end())
    {
      owner_frame_depth = top.owner_frame_depth;
      return true;
    }
  }

  return false;
}

bool goto_symext::enforce_exception_specifications(
  const std::vector<irep_idt> &exception_list,
  bool caught,
  std::size_t handler_owner_depth,
  bool &dispatch_result)
{
  // A frame allows the exception if any of the thrown ids (the dynamic type
  // and its bases) is permitted by the frame's specification.
  auto frame_allows = [&exception_list](const exception_specificationt &spec) {
    if (!spec.is_restrictive())
      return true;
    for (const auto &id : exception_list)
      if (spec.allows(id))
        return true;
    return false;
  };

  // The exception exits every frame between the throw point (the top frame)
  // and the frame that catches it. If it is never caught, it exits all frames.
  // The handler's own frame is entered, not exited, so it is not checked.
  const std::size_t top_depth = cur_state->call_stack.size() - 1;
  const std::size_t lowest_exited = caught ? handler_owner_depth + 1 : 0;

  // Enforce the innermost violated specification first.
  for (std::size_t depth = top_depth + 1; depth-- > lowest_exited;)
  {
    const exception_specificationt &spec =
      cur_state->call_stack[depth].exception_spec;
    if (frame_allows(spec))
      continue;

    if (spec.kind == exception_specificationt::kindt::non_throwing)
    {
      // noexcept boundary crossed by an exception: std::terminate.
      if (!terminate_handler())
      {
        claim(
          gen_false_expr(),
          "An exception escapes a noexcept function; std::terminate is "
          "called.");
        cur_state->guard.make_false();
      }
      dispatch_result = false;
      return true;
    }

    // Legacy dynamic specification throw(T...) violated: std::unexpected.
    if (!unexpected_handler())
    {
      std::string msg =
        "Trying to throw an exception but it's not allowed by declaration.\n\n";
      msg += "  Exception type: " + exception_list.begin()->as_string();
      msg += "\n  Allowed exceptions:";
      for (const auto &allowed : spec.allowed_types)
        msg += "\n   - " + allowed.as_string();
      claim(gen_false_expr(), msg);
      dispatch_result = true;
      return true;
    }
    dispatch_result = false;
    return true;
  }

  return false;
}

bool goto_symext::symex_throw_dispatch(const expr2tc &throw_code)
{
  irep_idt catch_name = "missing";
  const goto_programt::const_targett *catch_insn = nullptr;
  const code_cpp_throw2t &throw_ref = to_code_cpp_throw2t(throw_code);
  const std::vector<irep_idt> &exceptions_thrown = throw_ref.exception_list;

  // Determine whether the innermost active catch region handles this throw and,
  // if so, which function frame owns that handler.
  std::size_t handler_owner_depth = 0;
  bool caught = exception_caught_in_top(exceptions_thrown, handler_owner_depth);

  // Enforce the exception specification of every frame the exception exits
  // before being caught (all frames if it is never caught). A restrictive
  // specification violated at a frame boundary triggers std::terminate (for a
  // non-throwing spec) or std::unexpected (for a legacy dynamic spec), and the
  // exception is never delivered to the handler.
  bool dispatch_result = false;
  if (enforce_exception_specifications(
        exceptions_thrown, caught, handler_owner_depth, dispatch_result))
    return dispatch_result;

  // No specification was violated; fall back to normal handler search.
  // We check before iterate over the throw list to save time:
  // If there is no catch, we return an error
  if (!cur_state->stack_catch.size())
  {
    if (!unexpected_handler())
    {
      if (!terminate_handler())
      {
        // An un-caught exception. Error
        const std::string &msg = "Throwing an exception of type " +
                                 exceptions_thrown.begin()->as_string() +
                                 " but there is not catch for it.";
        claim(gen_false_expr(), msg);
        return true;
      }
    }
    return false;
  }

  // Get the list of catchs
  goto_symex_statet::exceptiont *except = &cur_state->stack_catch.top();

  // It'll be used for catch ordering when throwing
  // a derived object with multiple inheritance
  unsigned old_id_number = -1, new_id_number = 0;

  goto_symex_statet::call_stackt old_stack = cur_state->call_stack;
  for (auto const &it : throw_ref.exception_list)
  {
    // Search for a catch with a matching type (including base classes)
    goto_symex_statet::exceptiont::catch_mapt::const_iterator c_it =
      except->catch_map.find(it);

    // Track which catch type was matched (might be a base class)
    irep_idt matched_catch_type = it;

    // If no exact match, check inheritance hierarchy for Python exceptions
    if (c_it == except->catch_map.end())
    {
      for (const auto &catch_entry : except->catch_map)
      {
        if (is_python_exception_subtype(it, catch_entry.first))
        {
          c_it = except->catch_map.find(catch_entry.first);
          matched_catch_type = catch_entry.first; // Track the actual catch type
          break;
        }
      }
    }

    // Do we have a catch for it?
    if (c_it != except->catch_map.end())
    {
      // We do!

      // Get current catch number and update if needed
      // Use matched_catch_type instead of it for the lookup
      new_id_number = (*except->catch_order.find(matched_catch_type)).second;

      if (new_id_number < old_id_number)
      {
        // Only restore call_stack when re-selecting a better catch handler.
        // Skip restoration on first match (old_id_number == -1) to avoid unnecessary
        // deep copy that causes crashes on macOS with Python exceptions.
        if (old_id_number != (unsigned)-1)
          cur_state->call_stack = old_stack;
        cur_state->guard.make_true();

        update_throw_target(except, c_it->second, throw_code);
        catch_insn = &c_it->second;
        catch_name = c_it->first;
      }

      // Save old number id
      old_id_number = new_id_number;
    }
    else // We don't have a catch for it
    {
      // If it's a pointer, we must look for a catch(void*)
      if (it.as_string().find("_ptr") != std::string::npos)
      {
        // It's a pointer!

        // Do we have an void*?
        c_it = except->catch_map.find("void_ptr");

        if (c_it != except->catch_map.end())
        {
          // Make the jump to void*
          update_throw_target(except, c_it->second, throw_code);
          catch_insn = &c_it->second;
          catch_name = c_it->first;
        }
      }
      else
      {
        // Do we have an ellipsis?
        c_it = except->catch_map.find("ellipsis");

        if (c_it != except->catch_map.end())
        {
          update_throw_target(except, c_it->second, throw_code, true);
          catch_insn = &c_it->second;
          catch_name = c_it->first;
        }
      }
    }
  }

  if (catch_insn == nullptr)
  {
    // No catch for type, void, or ellipsis
    // Call terminate handler before showing error message
    if (!terminate_handler())
    {
      // An un-caught exception. Error
      const std::string &msg = "Throwing an exception of type " +
                               exceptions_thrown.begin()->as_string() +
                               " but there is not catch for it.";
      claim(gen_false_expr(), msg);
      // Ensure no further execution along this path.
      cur_state->guard.make_false();
    }

    return false;
  }

  log_debug(
    "symex",
    "Caught by catch({}) at file {} line {}",
    catch_name,
    (*catch_insn)->location.file(),
    (*catch_insn)->location.line());

  return true;
}

bool goto_symext::symex_throw()
{
  const goto_programt::instructiont &instruction = *cur_state->source.pc;
  const code_cpp_throw2t &throw_ref = to_code_cpp_throw2t(instruction.code);

  if (handle_rethrow(throw_ref.operand, instruction))
    return true;

  last_throw = const_cast<goto_programt::instructiont *>(&instruction);

  log_debug(
    "symex",
    "Exception thrown of type {} at file {} line {}",
    throw_ref.exception_list.begin()->as_string(),
    instruction.location.file(),
    instruction.location.line());

  return symex_throw_dispatch(instruction.code);
}

bool goto_symext::symex_throw_bad_cast()
{
  // Reuse the previously constructed instruction if available.
  if (!bad_cast_throw.code)
  {
    // Depending on the Clang/LLVM version the <typeinfo> model's class is
    // recorded either with the un-elaborated name ("tag-std::bad_cast") or with
    // the elaborated one ("tag-class std::bad_cast"); newer LLVM uses the
    // latter, which must match the name the catch clause carries. Try both.
    const symbolt *bad_cast_sym = ns.lookup("tag-std::bad_cast");
    if (!bad_cast_sym)
      bad_cast_sym = ns.lookup("tag-class std::bad_cast");
    if (!bad_cast_sym)
    {
      // <typeinfo> not included — emit a hard failure directly.
      claim(
        gen_false_expr(),
        "dynamic_cast<T&> failed; include <typeinfo> for std::bad_cast");
      return true;
    }

    // Build exception_list: concrete type + all direct base classes,
    // stripping the "tag-" prefix to match the catch-side convention.
    std::vector<irep_idt> exception_list;
    const std::string type_id = id2string(bad_cast_sym->id);
    exception_list.emplace_back(type_id.substr(4)); // "std::bad_cast"
    if (bad_cast_sym->get_type().id() == "struct")
    {
      const struct_typet &st = to_struct_type(bad_cast_sym->get_type());
      const exprt &bases = static_cast<const exprt &>(st.find("bases"));
      if (bases.is_not_nil())
        for (const auto &base : bases.get_sub())
          exception_list.emplace_back(id2string(base.id()).substr(4));
    }

    // Build a nondet operand of bad_cast type for the thrown object.
    type2tc bad_cast_type = migrate_symbol_type(*bad_cast_sym);
    expr2tc nondet_op = sideeffect2tc(
      bad_cast_type,
      expr2tc(),
      expr2tc(),
      std::vector<expr2tc>(),
      type2tc(),
      sideeffect2t::allockind::nondet);
    replace_nondet(nondet_op);

    bad_cast_throw.make_throw();
    bad_cast_throw.code = code_cpp_throw2tc(nondet_op, exception_list);
    last_throw = &bad_cast_throw;
  }

  log_debug("symex", "dynamic_cast<T&> failure: throwing std::bad_cast");

  return symex_throw_dispatch(bad_cast_throw.code);
}

bool goto_symext::terminate_handler()
{
  // We must look on the context if the user included exception lib
  const symbolt *tmp = ns.lookup("std::terminate()");
  bool is_included = !tmp;

  // If it do, we must call the terminate function:
  // It'll call the current function handler
  if (!is_included)
  {
    codet terminate_function = to_code(tmp->get_value().op0());

    // We only call it if the user replaced the default one
    if (terminate_function.op1().identifier() == "std::default_terminate()")
      return false;

    // Call the function
    expr2tc da_funk;
    migrate_expr(terminate_function, da_funk);
    symex_function_call(da_funk);
    return true;
  }

  // If it wasn't included, we do nothing. The error message will be
  // shown to the user as there is a throw without catch.
  return false;
}

bool goto_symext::unexpected_handler()
{
  // Look if we already on the unexpected flow
  // If true, we shouldn't call the unexpected handler again
  if (inside_unexpected)
    return false;

  // We must look on the context if the user included exception lib
  const symbolt *tmp = ns.lookup("c:@N@std@F@unexpected#");
  bool is_included = !tmp;

  // If it do, we must call the unexpected function:
  // It'll call the current function handler
  if (!is_included)
  {
    // We only call it if the user replaced the default one
    const symbolt *handler = ns.lookup("c:@F@__ESBMC_unexpected");
    if (!handler)
      return false;

    expr2tc the_call;
    code_function_callt unexpected_function;
    unexpected_function.function() = handler->get_value();
    migrate_expr(unexpected_function, the_call);

    // Indicate there we're inside the unexpected flow
    inside_unexpected = true;

    // Call the function
    symex_function_call(the_call);
    unexpected_end = handler->get_value().identifier();
    return true;
  }

  // If it wasn't included, we do nothing. The error message will be
  // shown to the user as there is a throw without catch.
  return false;
}

void goto_symext::update_throw_target(
  goto_symex_statet::exceptiont *except [[maybe_unused]],
  goto_programt::const_targett target,
  const expr2tc &code,
  bool is_ellipsis)
{
  // Something is going to catch, therefore we need something to be caught.
  // Assign that something to a variable and make records so that it's merged
  // into the right place in the future.
  assert(!is_nil_expr(code));
  const code_cpp_throw2t &throw_insn = to_code_cpp_throw2t(code);

  // Generate a name to assign this to.
  expr2tc thrown_obj =
    symbol2tc(throw_insn.operand->type, irep_idt("symex_throw::thrown_obj"));
  expr2tc operand = throw_insn.operand;
  symex_assign(code_assign2tc(thrown_obj, operand));

  // Target is, as far as I can tell, always a declaration of the variable
  // that the thrown obj ends up in, and is followed by a (blank) assignment
  // to it. So point at the next insn.
  //
  // An ellipsis (catch-all) handler binds no exception variable: its target
  // points straight at the first instruction of the handler body, which may be
  // an ASSERT/ASSUME/GOTO whose `code` field is nil. Only inspect target->code
  // (and record the thrown object for the bound variable) when one exists.
  if (!is_ellipsis)
  {
    // Now record that value for future reference.
    if (!is_pointer_type(target->code->type))
      cur_state->rename(thrown_obj);

    assert(is_code_decl2t(target->code));
    target++;
    assert(is_code_assign2t(target->code));
    // signed int b;
    // b = NONDET(signed int); move to this line

    // Signal assignment code to fetch the thrown object and rewrite the
    // assignment, assigning the thrown obj to the local variable.
    thrown_obj_map.insert_or_assign(target, thrown_obj);
  }

  if (!options.get_bool_option("extended-try-analysis"))
  {
    // Search backwards through stack frames, looking for the frame that
    // contains the function containing the target instruction.
    goto_symex_statet::call_stackt::reverse_iterator i;
    for (i = cur_state->call_stack.rbegin(); i != cur_state->call_stack.rend();
         ++i)
    {
      irep_idt id = i->function_identifier.empty() ? "__ESBMC_main"
                                                   : i->function_identifier;
      if (id == target->function)
      {
        statet::merge_state_listt &merge_state_list =
          i->merge_state_map[target];

        merge_state_list.emplace_back(*cur_state);
        cur_state->guard.make_false();
        break;
      }
    }

    assert(
      (i != cur_state->call_stack.rend() ||
       target->function == "__ESBMC_main") &&
      "Target instruction in throw "
      "handler not in any function frame on the stack");
  }
}

bool goto_symext::handle_rethrow(
  const expr2tc &operand,
  const goto_programt::instructiont &instruction)
{
  // throw without argument, we must rethrow last exception
  if (is_nil_expr(operand))
  {
    if (
      last_throw != nullptr &&
      to_code_cpp_throw2t(last_throw->code).exception_list.size())
    {
      // get exception from last throw
      std::vector<irep_idt>::const_iterator e_it =
        to_code_cpp_throw2t(last_throw->code).exception_list.begin();

      // update current state exception list
      goto_programt::instructiont &mutable_ref =
        const_cast<goto_programt::instructiont &>(instruction);
      to_code_cpp_throw2t(mutable_ref.code).exception_list.push_back((*e_it));
      to_code_cpp_throw2t(mutable_ref.code).operand =
        to_code_cpp_throw2t(last_throw->code).operand;

      return false;
    }

    const std::string &msg = "Trying to re-throw without last exception.";
    claim(gen_false_expr(), msg);
    return true;
  }
  return false;
}
