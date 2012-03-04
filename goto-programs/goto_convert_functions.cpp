/*******************************************************************\

Module: Goto Programs with Functions

Author: Daniel Kroening

Date: June 2003

\*******************************************************************/

#include <assert.h>

#include <base_type.h>
#include <std_code.h>

#include "goto_convert_functions.h"
#include "goto_inline.h"

/*******************************************************************\

Function: goto_convert_functionst::goto_convert_functionst

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

goto_convert_functionst::goto_convert_functionst(
  contextt &_context,
  const optionst &_options,
  goto_functionst &_functions,
  message_handlert &_message_handler):
  goto_convertt(_context, _options, _message_handler),
  functions(_functions)
{
	if (options.get_bool_option("inlining"))
	  inlining=true;
	else
	  inlining=false;
}

/*******************************************************************\

Function: goto_convert_functionst::~goto_convert_functionst

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

goto_convert_functionst::~goto_convert_functionst()
{
}

/*******************************************************************\

Function: goto_convert_functionst::goto_convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convert_functionst::goto_convert()
{
  // warning! hash-table iterators are not stable

  typedef std::list<irep_idt> symbol_listt;
  symbol_listt symbol_list;

  forall_symbols(it, context.symbols)
  {
    if(!it->second.is_type && it->second.type.is_code())
      symbol_list.push_back(it->first);
  }

  for(symbol_listt::const_iterator
      it=symbol_list.begin();
      it!=symbol_list.end();
      it++)
  {
    convert_function(*it);
  }

  functions.compute_location_numbers();

  // inline those functions marked as "inlined"
  if (!inlining) {
    goto_partial_inline(
      functions,
      ns,
      get_message_handler());
  }
}

/*******************************************************************\

Function: goto_convert_functionst::hide

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_convert_functionst::hide(const goto_programt &goto_program)
{
  for(goto_programt::instructionst::const_iterator
      i_it=goto_program.instructions.begin();
      i_it!=goto_program.instructions.end();
      i_it++)
  {
    for(goto_programt::instructiont::labelst::const_iterator
        l_it=i_it->labels.begin();
        l_it!=i_it->labels.end();
        l_it++)
    {
      if(*l_it=="__ESBMC_HIDE")
        return true;
    }
  }

  return false;
}

/*******************************************************************\

Function: goto_convert_functionst::add_return

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convert_functionst::add_return(
  goto_functionst::goto_functiont &f,
  const locationt &location)
{
  if(!f.body.instructions.empty() &&
     f.body.instructions.back().is_return())
    return; // not needed, we have one already

  // see if we have an unconditional goto at the end
  if(!f.body.instructions.empty() &&
     f.body.instructions.back().is_goto() &&
     f.body.instructions.back().guard.is_true())
    return;

  goto_programt::targett t=f.body.add_instruction();
  t->make_return();
  t->code=code_returnt();
  t->location=location;

  exprt rhs=exprt("sideeffect", f.type.return_type());
  rhs.statement("nondet");
  t->code.move_to_operands(rhs);
}

/*******************************************************************\

Function: goto_convert_functionst::convert_function

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convert_functionst::convert_function(const irep_idt &identifier)
{
  goto_functionst::goto_functiont &f=functions.function_map[identifier];
  const symbolt &symbol=ns.lookup(identifier);

  // make tmp variables local to function
  tmp_symbol_prefix=id2string(symbol.name)+"::$tmp::";
  temporary_counter=0;

  f.type=to_code_type(symbol.type);
  f.body_available=symbol.value.is_not_nil();

  if(f.body_available)
  {
    const code_typet::argumentst &arguments=f.type.arguments();

    std::list<irep_idt> arg_ids;

    // add as local variables
    for(code_typet::argumentst::const_iterator
        it=arguments.begin();
        it!=arguments.end();
        it++)
    {
      const irep_idt &identifier=it->get_identifier();
      assert(identifier!="");
      arg_ids.push_back(identifier);
    }

    if(!symbol.value.is_code())
    {
      err_location(symbol.value);
      throw "got invalid code for function `"+id2string(identifier)+"'";
    }

    codet tmp(to_code(symbol.value));

    locationt end_location;

    if(to_code(symbol.value).get_statement()=="block")
      end_location=static_cast<const locationt &>(
        symbol.value.end_location());
    else
      end_location.make_nil();

    targets=targetst();
    targets.return_set=true;
    targets.return_value=
      f.type.return_type().id()!="empty" &&
      f.type.return_type().id()!="constructor" &&
      f.type.return_type().id()!="destructor";

    goto_convert_rec(tmp, f.body);

    // add non-det return value, if needed
    if(targets.return_value)
      add_return(f, end_location);

    // add end of function

    goto_programt::targett t=f.body.add_instruction();
    t->type=END_FUNCTION;
    t->location=end_location;
    t->code.identifier(identifier);

    if(to_code(symbol.value).get_statement()=="block")
      t->location=static_cast<const locationt &>(
        symbol.value.end_location());

    // do local variables
    Forall_goto_program_instructions(i_it, f.body)
    {
      i_it->add_local_variables(arg_ids);
      i_it->function=identifier;
    }

    f.body.compute_targets();
    f.body.number_targets();

    if(hide(f.body))
      f.type.hide(true);
  }

}

/*******************************************************************\

Function: goto_convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convert(
  codet &code,
  contextt &context,
  const optionst &options,
  goto_functionst &functions,
  message_handlert &message_handler)
{
}

/*******************************************************************\

Function: goto_convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convert(
  contextt &context,
  const optionst &options,
  goto_functionst &functions,
  message_handlert &message_handler)
{
  goto_convert_functionst goto_convert_functions(
    context, options, functions, message_handler);

  try
  {
    goto_convert_functions.thrash_type_symbols();
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

static void
fetch_type_dependancies(const irept &type, std::set<irep_idt> &deps)
{

  if (type.id() == "pointer")
    return;

  if (type.id() == "symbol") {
    if (type.identifier().as_string().find("$type") != std::string::npos) {
      deps.insert(type.identifier());
      return;
    }
  }

  forall_irep(it, type.get_sub())
    fetch_type_dependancies(*it, deps);

  forall_named_irep(it, type.get_named_sub())
    fetch_type_dependancies(it->second, deps);

  forall_named_irep(it, type.get_comments())
    fetch_type_dependancies(it->second, deps);

  return;
}

void
goto_convert_functionst::rename_types(irept &type)
{

  if (type.id() == "pointer")
    return;

  if (type.id() == "symbol") {
    if (type.identifier().as_string().find("$type") != std::string::npos) {
      symbolst::const_iterator it = context.symbols.find(type.identifier());
      assert(it != context.symbols.end());
      type = it->second.type;
      return;
    }
  }

  Forall_irep(it, type.get_sub())
    rename_types(*it);

  Forall_named_irep(it, type.get_named_sub())
    rename_types(it->second);

  Forall_named_irep(it, type.get_comments())
    rename_types(it->second);

  return;
}

void
goto_convert_functionst::wallop_type(irep_idt name,
                         std::map<irep_idt, std::set<irep_idt> > &typenames)
{

  // If this type doesn't depend on anything, no need to rename anything.
  std::set<irep_idt> &deps = typenames.find(name)->second;
  if (deps.size() == 0)
    return;

  // Iterate over our dependancies ensuring they're resolved.
  for (std::set<irep_idt>::iterator it = deps.begin(); it != deps.end(); it++)
    wallop_type(*it, typenames);

  // And finally perform renaming.
  symbolst::iterator it = context.symbols.find(name);
  rename_types(it->second.type);
  deps.clear();
  return;
}

void
goto_convert_functionst::thrash_type_symbols(void)
{

  // This function has one purpose: remove as many type symbols as possible.
  // This is easy enough by just following each type symbol that occurs and
  // replacing it with the value of the type name. However, if we have a pointer
  // in a struct to itself, this breaks down. Therefore, don't rename types of
  // pointers; they have a type already; they're pointers.

  // Start off by collecting all type symbols. Identified by having "$type"
  // in their names :|. And, compute their dependancies.

  std::map<irep_idt, std::set<irep_idt> > typenames;

  forall_symbols(it, context.symbols) {
    if (it->second.name.as_string().find("$type") != std::string::npos) {
      std::set<irep_idt> depset;
      fetch_type_dependancies(it->second.type, depset);
      typenames[it->second.name] = depset;
    }
  }

  // Now, repeatedly rename all types. When we encounter a type that contains
  // unresolved symbols, resolve it first, then include it into this type.
  // This means that we recurse to whatever depth of nested types the user
  // has. With at least a meg of stack, I doubt that's really a problem.
  std::map<irep_idt, std::set<irep_idt> >::iterator it;
  for (it = typenames.begin(); it != typenames.end(); it++)
    wallop_type(it->first, typenames);

  return;
}
