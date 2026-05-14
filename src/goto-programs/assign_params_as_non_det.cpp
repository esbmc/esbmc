#include <goto-programs/assign_params_as_non_det.h>

symbolt *assign_params_as_non_det::get_default_symbol(
  typet type,
  std::string name,
  std::string id,
  locationt location)
{
  symbolt symbol;
  symbol.location = std::move(location);
  symbol.type = std::move(type);
  symbol.name = name;
  symbol.id = id;

  symbol.static_lifetime = false;
  symbol.is_extern = false;
  symbol.lvalue = true;

  symbolt *new_sym;
  context.move(symbol, new_sym);
  return new_sym;
}

/// Build an irep2 symbol expression referring to @p sym.
static expr2tc sym_to_expr2tc(const symbolt &sym)
{
  return symbol2tc(migrate_type(sym.type), sym.id);
}

/// Append @p instr (with location and function copied from @p it) before @p it
/// in @p program, leaving @p it pointing at the same original instruction.
static void insert_before(
  goto_programt &program,
  goto_programt::targett &it,
  expr2tc code,
  goto_program_instruction_typet type,
  const locationt &loc)
{
  goto_programt::instructiont instr;
  instr.type = type;
  instr.code = std::move(code);
  instr.location = loc;
  instr.function = it->location.get_function();
  program.insert_swap(it++, instr);
  --it;
}

bool assign_params_as_non_det::runOnFunction(
  std::pair<const irep_idt, goto_functiont> &F)
{
  const symbolt *fn_sym = context.find_symbol(F.first);
  if (fn_sym == nullptr)
    return false; // Not exist

  if (!fn_sym->type.is_code())
    return false; // Not expected
  const code_typet t = to_code_type(fn_sym->type);

  if (fn_sym->name.as_string() != target_function)
    return false; // Not target function

  if (!F.second.body_available)
    return false; // Empty function

  /*
    Foreach parameter, create an assignment to nondet value and insert in the front
    E.g. func(int x, bool y) {...}
    =>
    func(int x, bool y)  { x = nondet_int(); y = nondet_bool(); ...}
  */
  goto_programt &goto_program = F.second.body;
  auto it = goto_program.instructions.begin();
  const locationt l = fn_sym->location;

  for (const auto &arg : t.arguments())
  {
    const auto &_id = arg.get("#identifier");
    const symbolt *lhs_sym = context.find_symbol(_id);
    if (lhs_sym == nullptr)
      return false; // Not expected
    expr2tc lhs = sym_to_expr2tc(*lhs_sym);
    type2tc l_t = lhs->type;

    insert_before(
      goto_program, it, code_assign2tc(lhs, gen_nondet(l_t)), ASSIGN, l);

    // additional handling for non-void pointer
    // so the pointer can point to an array (decay), which is covered above
    // also, the pointer can point to an object/variable, which is covered below
    if (!is_pointer_type(l_t))
      continue;
    const type2tc &subtype = to_pointer_type(l_t).subtype;
    if (is_empty_type(subtype))
      continue;

    /*
      e.g. int* lhs;
      to
      bool lhs#temp_bool;
      lhs#temp_bool = nondet();
      if(!lhs#temp_bool)
      {
        int lhs#temp;
        lhs#temp = nondet();
        assume(lhs#temp != 0);
        lhs = &lhs#temp;
      }
    */

    // lhs = nondet();  (initial null/nondet pointer assignment)
    insert_before(
      goto_program, it, code_assign2tc(lhs, gen_nondet(l_t)), ASSIGN, l);

    // resolve named subtypes for the temp object's symbol
    typet obj_typet = lhs_sym->type.subtype();
    if (obj_typet.is_symbol())
      obj_typet = context.find_symbol(obj_typet.identifier())->type;

    symbolt *obj_sym = get_default_symbol(
      obj_typet,
      lhs_sym->name.as_string() + "#temp",
      lhs_sym->id.as_string() + "#temp",
      l);

    symbolt *flag_sym = get_default_symbol(
      bool_typet(),
      lhs_sym->name.as_string() + "#temp_bool",
      lhs_sym->id.as_string() + "#temp_bool",
      l);

    // declare temp boolean variable: bool lhs#temp_bool;
    insert_before(
      goto_program,
      it,
      code_decl2tc(migrate_type(flag_sym->type), flag_sym->id),
      DECL,
      l);

    // lhs#temp_bool = nondet();
    expr2tc flag = sym_to_expr2tc(*flag_sym);
    insert_before(
      goto_program,
      it,
      code_assign2tc(flag, gen_nondet(flag->type)),
      ASSIGN,
      l);

    // declare temp object: T lhs#temp;
    insert_before(
      goto_program,
      it,
      code_decl2tc(migrate_type(obj_sym->type), obj_sym->id),
      DECL,
      l);

    // lhs#temp = nondet();
    expr2tc obj = sym_to_expr2tc(*obj_sym);
    insert_before(
      goto_program, it, code_assign2tc(obj, gen_nondet(obj->type)), ASSIGN, l);

    // assume((bool)lhs#temp);  (skipped: branch not counted in coverage)
    {
      goto_programt::instructiont assume;
      assume.make_assumption(typecast2tc(get_bool_type(), obj));
      assume.location = l;
      assume.location.property("skipped");
      assume.function = it->location.get_function();
      goto_program.insert_swap(it++, assume);
      --it;
    }

    // lhs = &lhs#temp;
    insert_before(
      goto_program,
      it,
      code_assign2tc(lhs, address_of2tc(obj->type, obj)),
      ASSIGN,
      l);

    // Wrap the four instructions just inserted (DECL flag, ASSIGN flag, DECL
    // obj, ASSIGN obj, ASSUME, ASSIGN addr) so they are guarded by the flag.
    //! hack: we use !flag (forward jump over the body) so the condition is not
    //! counted in goto_coverage.
    goto_programt::instructiont if_instr;
    if_instr.location = l;
    if_instr.location.property("skipped");
    if_instr.function = it->location.get_function();
    if_instr.make_goto(it);
    if_instr.guard = not2tc(flag);

    /* insert
       Instrument_1   <-- DECL flag
       Instrument_2   <-- ASSIGN flag
       Instrument_3   <-- DECL obj
       Instrument_4   <-- ASSIGN obj
       Instrument_5   <-- ASSUME
       Instrument_6   <-- ASSIGN lhs = &obj
       Origin_1       <-- 'it' is here; we want IF !flag GOTO Origin_1
                          inserted before Instrument_1
    */
    --it; // ASSIGN lhs = &obj
    --it; // ASSUME
    --it; // ASSIGN obj
    --it; // DECL obj
    --it; // ASSIGN flag
    --it; // DECL flag
    goto_program.insert_swap(it++, if_instr);
    ++it; // ASSIGN flag
    ++it; // DECL obj
    ++it; // ASSIGN obj
    ++it; // ASSUME
    ++it; // ASSIGN lhs = &obj
    goto_program.compute_target_numbers();
  }

  goto_program.update();
  return true;
}
