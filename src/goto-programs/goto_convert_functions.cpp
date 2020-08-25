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
    goto_convert_functions.thrash_type_symbols();
    goto_convert_functions.fixup_unions();
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

void goto_convert_functionst::collect_type(
  const irept &type,
  typename_sett &deps)
{
  if(type.id() == "pointer")
    return;

  if(type.id() == "symbol")
  {
    deps.insert(type.identifier());
    return;
  }

  collect_expr(type, deps);
}

void goto_convert_functionst::collect_expr(
  const irept &expr,
  typename_sett &deps)
{
  if(expr.id() == "pointer")
    return;

  forall_irep(it, expr.get_sub())
  {
    collect_expr(*it, deps);
  }

  forall_named_irep(it, expr.get_named_sub())
  {
    collect_type(it->second, deps);
  }

  forall_named_irep(it, expr.get_comments())
  {
    collect_type(it->second, deps);
  }
}

void goto_convert_functionst::rename_types(
  irept &type,
  const symbolt &cur_name_sym,
  const irep_idt &sname)
{
  if(type.id() == "pointer")
    return;

  // Some type symbols aren't entirely correct. This is because (in the current
  // 27_exStbFb test) some type symbols get the module name inserted into the
  // name -- so int32_t becomes main::int32_t.
  //
  // Now this makes entire sense, because int32_t could be something else in
  // some other file. However, because type symbols aren't squashed at type
  // checking time (which, you know, might make sense) we now don't know what
  // type symbol to link "int32_t" up to. So; instead we test to see whether
  // a type symbol is linked correctly, and if it isn't we look up what module
  // the current block of code came from and try to guess what type symbol it
  // should have.

  typet type2;
  if(type.id() == "symbol")
  {
    if(type.identifier() == sname)
    {
      // A recursive symbol -- the symbol we're about to link to is in fact the
      // one that initiated this chain of renames. This leads to either infinite
      // loops or segfaults, depending on the phase of the moon.
      // It should also never happen, but with C++ code it does, because methods
      // are part of the type, and methods can take a full struct/object as a
      // parameter, not just a reference/pointer. So, that's a legitimate place
      // where we have this recursive symbol dependancy situation.
      // The workaround to this is to just ignore it, and hope that it doesn't
      // become a problem in the future.
      return;
    }

    const symbolt *sym;
    if(!ns.lookup(type.identifier(), sym))
    {
      // If we can just look up the current type symbol, use that.
      type2 = ns.follow((typet &)type);
    }
    else
    {
      // Otherwise, try to guess the namespaced type symbol
      std::string ident =
        cur_name_sym.module.as_string() + type.identifier().as_string();

      // Try looking that up.
      if(!ns.lookup(irep_idt(ident), sym))
      {
        irept tmptype = type;
        tmptype.identifier(irep_idt(ident));
        type2 = ns.follow((typet &)tmptype);
      }
      else
      {
        // And if we fail
        std::cerr << "Can't resolve type symbol " << ident;
        std::cerr << " at symbol squashing time" << std::endl;
        abort();
      }
    }

    type = type2;
    return;
  }

  rename_exprs(type, cur_name_sym, sname);
}

void goto_convert_functionst::rename_exprs(
  irept &expr,
  const symbolt &cur_name_sym,
  const irep_idt &sname)
{
  if(expr.id() == "pointer")
    return;

  Forall_irep(it, expr.get_sub())
    rename_exprs(*it, cur_name_sym, sname);

  Forall_named_irep(it, expr.get_named_sub())
  {
    if(it->first == "type" || it->first == "subtype")
    {
      rename_types(it->second, cur_name_sym, sname);
    }
    else
    {
      rename_exprs(it->second, cur_name_sym, sname);
    }
  }

  Forall_named_irep(it, expr.get_comments())
    rename_exprs(it->second, cur_name_sym, sname);
}

void goto_convert_functionst::wallop_type(
  irep_idt name,
  std::map<irep_idt, std::set<irep_idt>> &typenames,
  const irep_idt &sname)
{
  // If this type doesn't depend on anything, no need to rename anything.
  std::set<irep_idt> &deps = typenames.find(name)->second;
  if(deps.size() == 0)
    return;

  // Iterate over our dependancies ensuring they're resolved.
  for(const auto &dep : deps)
    wallop_type(dep, typenames, sname);

  // And finally perform renaming.
  symbolt *s = context.find_symbol(name);
  rename_types(s->type, *s, sname);
  deps.clear();
}

void goto_convert_functionst::thrash_type_symbols()
{
  // This function has one purpose: remove as many type symbols as possible.
  // This is easy enough by just following each type symbol that occurs and
  // replacing it with the value of the type name. However, if we have a pointer
  // in a struct to itself, this breaks down. Therefore, don't rename types of
  // pointers; they have a type already; they're pointers.

  // Collect a list of all type names. This it required before this entire
  // thing has no types, and there's no way (in C++ converted code at least)
  // to decide what name is a type or not.
  typename_sett names;
  context.foreach_operand([this, &names](const symbolt &s) {
    collect_expr(s.value, names);
    collect_type(s.type, names);
  });

  // Try to compute their dependencies.

  typename_mapt typenames;
  context.foreach_operand([this, &names, &typenames](const symbolt &s) {
    if(names.find(s.id) != names.end())
    {
      typename_sett list;
      collect_expr(s.value, list);
      collect_type(s.type, list);
      typenames[s.id] = list;
    }
  });

  for(auto &it : typenames)
    it.second.erase(it.first);

  // Now, repeatedly rename all types. When we encounter a type that contains
  // unresolved symbols, resolve it first, then include it into this type.
  // This means that we recurse to whatever depth of nested types the user
  // has. With at least a meg of stack, I doubt that's really a problem.
  std::map<irep_idt, std::set<irep_idt>>::iterator it;
  for(it = typenames.begin(); it != typenames.end(); it++)
    wallop_type(it->first, typenames, it->first);

  // And now all the types have a fixed form, rename types in all existing code.
  context.Foreach_operand([this](symbolt &s) {
    rename_types(s.type, s, s.id);
    rename_exprs(s.value, s, s.id);
  });
}

void goto_convert_functionst::fixup_unions()
{
  // Iterate over all types and expressions, replacing:
  //  * Non-pointer union types with byte arrays of corresponding size
  //  * All union member accesses with the following pattern:
  //      dataobj.field => ((uniontype*)&dataobj)->field
  // Thus ensuring that all unions become byte arrays, and all accesses to
  // them _as_ unions get converted into byte array accesses at the pointer
  // dereference layer.

  context.Foreach_operand([this](symbolt &s) {
    fix_union_type(s.type, false);
    fix_union_expr(s.value);
  });
}

void goto_convert_functionst::fix_union_type(typet &type, bool is_pointer)
{
  if(!is_pointer && type.is_union())
  {
    // Replace with byte array. Must use migrated type though, because we need
    // one authorative type_byte_size function
    type2tc new_type;
    migrate_type(type, new_type);
    auto size = type_byte_size(new_type);
    new_type = type2tc(
      new array_type2t(get_uint8_type(), gen_ulong(size.to_uint64()), false));
    type = migrate_type_back(new_type);
    return;
  }

  // Otherwise, recurse, taking care to handle pointers appropriately. All
  // pointers to unions should remain union types.
  if(type.is_pointer())
  {
    fix_union_type(type.subtype(), true);
  }
  else
  {
    Forall_irep(it, type.get_sub())
      fix_union_type((typet &)*it, false);
    Forall_named_irep(it, type.get_named_sub())
      fix_union_type((typet &)it->second, false);
  }
}

void goto_convert_functionst::fix_union_expr(exprt &expr)
{
  // We care about one kind of expression: member expressions that access a
  // union field. We also need to rewrite types as we come across them.
  if(expr.is_member())
  {
    // Are we accessing a union? If it's already a dereference, that's fine.
    if(expr.op0().type().is_union() && !expr.op0().is_dereference())
    {
      // Rewrite 'dataobj.field' to '((uniontype*)&dataobj)->field'
      expr2tc dataobj;
      migrate_expr(expr.op0(), dataobj);
      type2tc union_type = dataobj->type;
      auto size = type_byte_size(union_type);
      type2tc array_type = type2tc(
        new array_type2t(get_uint8_type(), gen_ulong(size.to_uint64()), false));
      type2tc union_pointer(new pointer_type2t(union_type));

      address_of2tc addrof(array_type, dataobj);
      typecast2tc cast(union_pointer, addrof);
      dereference2tc deref(union_type, cast);
      expr.op0() = migrate_expr_back(deref);

      // Fix type -- it needs to remain a union at the top level
      fix_union_type(expr.type(), false);
      fix_union_expr(expr.op0());
    }
    else
    {
      Forall_operands(it, expr)
        fix_union_expr(*it);
      fix_union_type(expr.type(), false);
    }
  }
  else if(expr.is_union())
  {
    // There may be union types embedded within this type; those need their
    // types fixing too.
    Forall_operands(it, expr)
      fix_union_expr(*it);
    fix_union_type(expr.type(), false);

    // A union expr is a constant/literal union. This needs to be flattened
    // out at this stage. Handle this by migrating immediately (which will
    // eliminate anything on union type), and overwriting this expression.
    expr2tc new_expr;
    migrate_expr(expr, new_expr);
    expr = migrate_expr_back(new_expr);
  }
  else if(expr.is_dereference())
  {
    // We want the dereference of a union pointer to evaluate to a union type,
    // as that can be picked apart by the pointer handling code. However, do
    // rewrite types if it points at a struct that contains a union, because
    // the correct type of a struct reference with a union is it, has it's
    // fields rewritten to be arrays. Actual accesses to that union field will
    // be transformed into a dereference to one of the fields _within_ the
    // union, so we never up constructing a union reference.
    Forall_operands(it, expr)
      fix_union_expr(*it);
    fix_union_type(expr.type(), true);
  }
  else
  {
    // Default action: recurse and beat types.

    fix_union_type(expr.type(), false);

    Forall_operands(it, expr)
      fix_union_expr(*it);
  }
}
