#include <cassert>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_inline.h>
#include <goto-programs/remove_no_op.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/i2string.h>
#include <util/prefix.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/type_byte_size.h>

#define IMPLIES(a, b) (!(a) || (b))

goto_convert_functionst::goto_convert_functionst(
  contextt &_context,
  optionst &_options,
  goto_functionst &_functions)
  : goto_convertt(_context, _options), functions(_functions)
{
}

void goto_convert_functionst::goto_convert()
{
  // warning! hash-table iterators are not stable

  symbol_listt symbol_list;
  context.Foreach_operand_in_order([&symbol_list](symbolt &s) {
    if (!s.is_type && s.type.is_code())
      symbol_list.push_back(&s);
  });

  for (auto &it : symbol_list)
  {
    convert_function(*it);
  }

  functions.compute_location_numbers();
}

bool goto_convert_functionst::hide(const goto_programt &goto_program)
{
  for (const auto &instruction : goto_program.instructions)
  {
    for (const auto &label : instruction.labels)
    {
      if (label == "__ESBMC_HIDE")
        return true;
    }
  }

  return false;
}

void goto_convert_functionst::add_return(
  goto_functiont &f,
  const locationt &location)
{
  if (!f.body.instructions.empty() && f.body.instructions.back().is_return())
    return; // not needed, we have one already

  // see if we have an unconditional goto at the end
  if (
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
  // Note: can be removed probably? as the new clang-cpp-frontend should've
  // done a pretty good job at resolving template overloading
  if (
    symbol.value.get("#speculative_template") == "1" &&
    symbol.value.get("#template_in_use") != "1")
    return;

  // make tmp variables local to function
  tmp_symbol = symbol_generator(id2string(symbol.id) + "::$tmp::");

  auto it = functions.function_map.find(identifier);
  if (it == functions.function_map.end())
    functions.function_map.emplace(identifier, goto_functiont());

  goto_functiont &f = functions.function_map.at(identifier);
  f.type = to_code_type(symbol.type);
  f.body_available = symbol.value.is_not_nil();

  if (!f.body_available)
    return;

  if (!symbol.value.is_code())
  {
    log_error("got invalid code for function `{}'", id2string(identifier));
    abort();
  }

  const codet &code = to_code(symbol.value);

  locationt end_location;

  if (to_code(symbol.value).get_statement() == "block")
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
  if (targets.has_return_value)
    add_return(f, end_location);

  // Wrap the body of functions name __VERIFIER_atomic_* with atomic_begin
  // and atomic_end
  if (
    !f.body.instructions.empty() &&
    has_prefix(id2string(identifier), "c:@F@__VERIFIER_atomic_"))
  {
    goto_programt::instructiont a_begin;
    a_begin.make_atomic_begin();
    a_begin.location = f.body.instructions.front().location;
    f.body.insert_swap(f.body.instructions.begin(), a_begin);

    goto_programt::targett a_end = f.body.add_instruction();
    a_end->make_atomic_end();
    a_end->location = end_location;

    Forall_goto_program_instructions (i_it, f.body)
    {
      if (i_it->is_goto() && i_it->targets.front()->is_end_function())
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

  if (config.ansi_c.cheri)
  {
    // Hide the cheri ptr compressed and decompressed traces
    const irep_idt &n = symbol.location.get_file();
    if (has_suffix(n, "cheri_compressed_cap_common.h"))
      f.body.hide = true;
  }

  if (hide(f.body))
    f.body.hide = true;
}

void goto_convert(
  contextt &context,
  optionst &options,
  goto_functionst &functions)
{
  goto_convert_functionst goto_convert_functions(context, options, functions);

  goto_convert_functions.thrash_type_symbols();
  goto_convert_functions.goto_convert();
}

void goto_convert_functionst::visit_sub_type(
  irept &type,
  std::set<irept *> &to_replace)
{
  // Subtypes of pointers that are symbols are not replaced.
  if (type.id() == "pointer")
    return;

  if (type.id() == "symbol")
  {
    to_replace.insert(&type);
    return;
  }

  visit_irept(type, to_replace);
}

void goto_convert_functionst::visit_irept(
  irept &irept_val,
  std::set<irept *> &to_replace)
{
  Forall_irep (it, irept_val.get_sub())
  {
    visit_irept(*it, to_replace);
  }
  auto handle_named_subt = [this, &to_replace](irept::named_subt &sub) {
    Forall_named_irep (it, sub)
    {
      if (it->first == "type" || it->first == "subtype")
      {
        visit_sub_type(it->second, to_replace);
      }
      else
      {
        visit_irept(it->second, to_replace);
      }
    }
  };
  handle_named_subt(irept_val.get_named_sub());
  handle_named_subt(irept_val.get_comments());
}

bool goto_convert_functionst::ensure_type_is_complete(typet &type)
{
  bool changed = false;
  if (type.is_struct() || type.is_union())
  {
    for (auto &comp : to_struct_union_type(type).components())
    {
      changed |= ensure_type_is_complete(comp.type());
    }
  }
  else if (type.is_array() || type.is_vector())
  {
    changed |= ensure_type_is_complete(type.subtype());
  }
  else if (
    type.is_signedbv() || type.is_unsignedbv() || type.is_fixedbv() ||
    type.is_floatbv() || type.is_pointer() || type.is_bool())
  {
    // already complete
  }
  else if (type.is_symbol())
  {
    const auto &symbol = to_symbol_type(type);
    symbolt *s = context.find_symbol(symbol.get_identifier());
    assert(s);
    assert(s->is_type);
    ensure_type_is_complete(s->type);
    type = s->type;
    changed |= true;
  }
  else
  {
    log_error("Unexpected type: {}", type.pretty());
    abort();
  }
  return changed;
}

void goto_convert_functionst::thrash_type_symbols()
{
  // 1. Ensure that all types in the context are complete.

  context.Foreach_operand([this](symbolt &s) {
    if (s.is_type)
    {
      // if the type has no "incomplete" flag, it should be complete
      if (!s.type.get_bool(irept::a_incomplete))
      {
        ensure_type_is_complete(s.type);
        assert(!ensure_type_is_complete(s.type)); // should be complete now
      }
    }
  });

  // 2. Visit all irepts to collect all symbol types that are not pointers to symbols.
  // (Pointers to symbols are not replaced, because we would get problems with recursive types (via pointers).)
  // We have to collect them first, because we cannot replace them while iterating over the context.
  // This is because it would lead to infinite recursion in the case of recursive types.
  std::set<irept *> to_replace;
  context.Foreach_operand([this, &to_replace](symbolt &s) {
    visit_irept(s.value, to_replace);
    visit_sub_type(s.type, to_replace);
  });

  // 3. Replace the collected symbol types with the type they reference.
  for (const auto type : to_replace)
  {
    assert(!type->identifier().empty());
    symbolt *s = context.find_symbol(type->identifier());
    assert(s);
    assert(s->is_type);
    assert(IMPLIES(
      !s->type.get_bool(irept::a_incomplete),
      !ensure_type_is_complete(s->type)));
    *type = s->type;
  }
}
