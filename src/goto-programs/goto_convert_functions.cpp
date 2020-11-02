/*******************************************************************\

Module: Goto Programs with Functions

Author: Daniel Kroening

Date: June 2003

\*******************************************************************/

#include <cassert>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_inline.h>
#include <goto-programs/remove_skip.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/i2string.h>
#include <util/prefix.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/type_byte_size.h>

goto_convert_functionst::goto_convert_functionst(
  contextt &_context,
  optionst &_options,
  goto_functionst &_functions,
  message_handlert &_message_handler)
  : goto_convertt(_context, _options, _message_handler), functions(_functions)
{
}

void goto_convert_functionst::goto_convert()
{
  // warning! hash-table iterators are not stable

  symbol_listt symbol_list;
  context.Foreach_operand_in_order([&symbol_list](symbolt &s) {
    if(!s.is_type && s.type.is_code())
      symbol_list.push_back(&s);
  });

  for(auto &it : symbol_list)
  {
    convert_function(*it);
  }

  functions.compute_location_numbers();
}

bool goto_convert_functionst::hide(const goto_programt &goto_program)
{
  for(const auto &instruction : goto_program.instructions)
  {
    for(const auto &label : instruction.labels)
    {
      if(label == "__ESBMC_HIDE")
        return true;
    }
  }

  return false;
}

void goto_convert_functionst::add_return(
  goto_functiont &f,
  const locationt &location)
{
  if(!f.body.instructions.empty() && f.body.instructions.back().is_return())
    return; // not needed, we have one already

  // see if we have an unconditional goto at the end
  if(
    !f.body.instructions.empty() && f.body.instructions.back().is_goto() &&
    is_true(f.body.instructions.back().guard))
    return;

  goto_programt::targett t = f.body.add_instruction();
  t->make_return();
  t->location = location;

  const typet &thetype = (f.type.return_type().id() == "symbol")
                           ? ns.follow(f.type.return_type())
                           : f.type.return_type();
  exprt rhs = exprt("sideeffect", thetype);
  rhs.statement("nondet");

  expr2tc tmp_expr;
  migrate_expr(rhs, tmp_expr);
  t->code = code_return2tc(tmp_expr);
}

void goto_convert_functionst::convert_function(symbolt &symbol)
{
  irep_idt identifier = symbol.id;

  // Apply a SFINAE test: discard unused C++ templates.
  if(
    symbol.value.get("#speculative_template") == "1" &&
    symbol.value.get("#template_in_use") != "1")
    return;

  // make tmp variables local to function
  tmp_symbol_prefix = id2string(symbol.id) + "::$tmp::";
  temporary_counter = 0;

  goto_functiont &f = functions.function_map[identifier];
  f.type = to_code_type(symbol.type);
  f.body_available = symbol.value.is_not_nil();

  if(!f.body_available)
    return;

  if(!symbol.value.is_code())
  {
    err_location(symbol.value);
    throw "got invalid code for function `" + id2string(identifier) + "'";
  }

  const codet &code = to_code(symbol.value);

  locationt end_location;

  if(to_code(symbol.value).get_statement() == "block")
    end_location =
      static_cast<const locationt &>(to_code_block(code).end_location());
  else
    end_location.make_nil();

  // add "end of function"
  goto_programt tmp_end_function;
  goto_programt::targett end_function = tmp_end_function.add_instruction();
  end_function->type = END_FUNCTION;
  end_function->location = end_location;

  targets = targetst();
  targets.set_return(end_function);
  targets.has_return_value = f.type.return_type().id() != "empty" &&
                             f.type.return_type().id() != "constructor" &&
                             f.type.return_type().id() != "destructor";

  goto_convert_rec(code, f.body);

  // add non-det return value, if needed
  if(targets.has_return_value)
    add_return(f, end_location);

  // Wrap the body of functions name __VERIFIER_atomic_* with atomic_bengin
  // and atomic_end
  if(
    !f.body.instructions.empty() &&
    has_prefix(id2string(identifier), "__VERIFIER_atomic_"))
  {
    goto_programt::instructiont a_begin;
    a_begin.make_atomic_begin();
    a_begin.location = f.body.instructions.front().location;
    f.body.insert_swap(f.body.instructions.begin(), a_begin);

    goto_programt::targett a_end = f.body.add_instruction();
    a_end->make_atomic_end();
    a_end->location = end_location;

    Forall_goto_program_instructions(i_it, f.body)
    {
      if(i_it->is_goto() && i_it->targets.front()->is_end_function())
      {
        i_it->targets.clear();
        i_it->targets.push_back(a_end);
      }
    }
  }

  // add "end of function"
  f.body.destructive_append(tmp_end_function);

  // do function tags (they are empty at this point)
  f.update_instructions_function(identifier);

  f.body.update();

  if(hide(f.body))
    f.body.hide = true;
}

void goto_convert(
  contextt &context,
  optionst &options,
  goto_functionst &functions,
  message_handlert &message_handler)
{
  goto_convert_functionst goto_convert_functions(
    context, options, functions, message_handler);

  try
  {
    goto_convert_functions.goto_convert();
  }

  catch(int)
  {
    goto_convert_functions.error();
  }

  catch(const char *e)
  {
    goto_convert_functions.error(e);
  }

  catch(const std::string &e)
  {
    goto_convert_functions.error(e);
  }

  if(goto_convert_functions.get_error_found())
    throw 0;
}
