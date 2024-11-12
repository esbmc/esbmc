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

bool assign_params_as_non_det::runOnFunction(
  std::pair<const dstring, goto_functiont> &F)
{
  if (context.find_symbol(F.first) == nullptr)
    return false; // Not exist

  exprt symbol = symbol_expr(*context.find_symbol(F.first));

  if (!symbol.type().is_code())
    return false; // Not expected
  code_typet t = to_code_type(symbol.type());

  if (symbol.name().as_string() != target_function)
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
  auto it = (goto_program).instructions.begin();
  locationt l = context.find_symbol(F.first)->location;

  for (const auto &arg : t.arguments())
  {
    // lhs
    const auto &_id = arg.get("#identifier");
    if (context.find_symbol(_id) == nullptr)
      return false; // Not expected
    exprt lhs = symbol_expr(*context.find_symbol(_id));

    typet l_t = lhs.type();
    if (l_t.is_pointer() && l_t.subtype() != empty_typet())
    {
      /*
        e.g. int* lhs;
        to
        lhs = null;
        bool temp;
        temp = nondet();
        if(temp)
        {
          int temp2;
          temp2 = nondet();
          assume(temp2 != 0);
          lhs = &temp2;
        }
        
        During this process we create two auxiliary variable. One for the boolean flag, and one for the object.
        - We need to consider the situation where pointers can be null, thus we put the assignment under an if-statement
        - If the flag (`temp`) is true, we create an object (`temp2`) and assign its address to the pointer.
      */

      // lhs = null;
      exprt zero_rhs = gen_zero(l_t);
      zero_rhs.location() = l;

      // assignment
      goto_programt zero_program;
      goto_programt::targett zero_assignment =
        zero_program.add_instruction(ASSIGN);
      zero_assignment->location = l;
      zero_assignment->function = it->location.get_function();

      code_assignt zero_assign(lhs, zero_rhs);
      zero_assign.location() = it->location;
      expr2tc new_zero_assign;
      migrate_expr(zero_assign, new_zero_assign);
      zero_assignment->code = new_zero_assign;

      // insert
      goto_program.insert_swap(it++, *zero_assignment);
      --it;

      // get subType() => int
      typet subt = l_t.subtype();
      // if it's symbol, get the original type
      if (subt.is_symbol())
        subt = context.find_symbol(subt.identifier())->type;

      // create obj and move it to the symbol table
      symbolt *new_sym = get_default_symbol(
        subt,
        lhs.name().as_string() + "#temp",
        lhs.identifier().as_string() + "#temp",
        l);

      code_declt _decl(symbol_expr(*new_sym));

      // declare temp boolean variable
      exprt _rhs = exprt("sideeffect", bool_typet());
      _rhs.statement("nondet");
      _rhs.location() = l;

      symbolt *new_sym2 = get_default_symbol(
        bool_typet(),
        lhs.name().as_string() + "#temp_bool",
        lhs.identifier().as_string() + "#temp_bool",
        l);
      code_declt _decl2(symbol_expr(*new_sym2));
      goto_programt tmp2;
      goto_programt::targett decl_statement2 = tmp2.add_instruction(DECL);
      decl_statement2->location = l;
      decl_statement2->function = it->location.get_function();
      expr2tc new_decl2;
      migrate_expr(_decl2, new_decl2);
      decl_statement2->code = new_decl2;
      goto_program.insert_swap(it++, *decl_statement2);
      --it;

      // assign nondet_bool
      goto_programt bool_tmp;
      goto_programt::targett bool_assignment = bool_tmp.add_instruction(ASSIGN);
      bool_assignment->location = l;
      bool_assignment->function = it->location.get_function();
      code_assignt bool_assign(symbol_expr(*new_sym2), _rhs);
      bool_assign.location() = l;
      expr2tc new_bool_assign;
      migrate_expr(bool_assign, new_bool_assign);
      bool_assignment->code = new_bool_assign;
      goto_program.insert_swap(it++, *bool_assignment);
      --it;

      // create a goto_instructiont DECL and insert to the original program  => DECL _temp
      goto_programt decl_tmp;
      goto_programt::targett decl_statement = decl_tmp.add_instruction(DECL);
      decl_statement->location = l;
      decl_statement->function = it->location.get_function();
      expr2tc new_decl;
      migrate_expr(_decl, new_decl);
      decl_statement->code = new_decl;

      // insert
      goto_program.insert_swap(it++, *decl_statement);
      --it;

      // set value of _temp; => temp = nondet
      _rhs = exprt("sideeffect", subt);
      _rhs.statement("nondet");
      _rhs.location() = l;

      // assignment
      goto_programt assign_tmp;
      goto_programt::targett assignment = assign_tmp.add_instruction(ASSIGN);
      assignment->location = l;
      assignment->function = it->location.get_function();

      code_assignt code_assign(symbol_expr(*new_sym), _rhs);
      code_assign.location() = l;
      expr2tc new_assign_expr;
      migrate_expr(code_assign, new_assign_expr);
      assignment->code = new_assign_expr;
      goto_program.insert_swap(it++, *assignment);
      --it;

      // do assume
      exprt n_lhs = typecast_exprt(symbol_expr(*new_sym), bool_type());
      expr2tc n_guard;
      migrate_expr(n_lhs, n_guard);
      goto_programt::instructiont instruction;
      instruction.make_assumption(n_guard);
      instruction.location = l;
      instruction.function = it->location.get_function();
      goto_program.insert_swap(it++, instruction);
      --it;

      // do assignment => lhs = &_temp
      goto_programt obj_address;
      goto_programt::targett addr_assignment =
        obj_address.add_instruction(ASSIGN);
      addr_assignment->location = l;
      addr_assignment->function = it->location.get_function();

      exprt _addr = address_of_exprt(symbol_expr(*new_sym));
      code_assignt addr_assign(lhs, _addr);
      addr_assign.location() = l;
      expr2tc new_addr_assign;
      migrate_expr(addr_assign, new_addr_assign);
      addr_assignment->code = new_addr_assign;

      goto_program.insert_swap(it++, *addr_assignment);
      --it;

      // create if statement => if(nondet_bool()) lhs = &_temp;
      //! hack: we do not do reverse !(nondet_bool)
      //! such that this condition will not be counted in the goto_coverage
      expr2tc guard;
      migrate_expr(symbol_expr(*new_sym2), guard);

      // inser if statement to the goto program
      goto_programt tmp4;
      goto_programt::targett if_statement = tmp4.add_instruction();
      if_statement->location = l;
      if_statement->function = it->location.get_function();

      if_statement->make_goto(it);
      if_statement->guard = guard;

      /* insert      
       Instrument_1
       Instrument_2  <-- we want to insert a statement before here
       Instrument_3
       Instrument_4  
       Origin_1        <-- 'it' is currently here
      */
      --it;
      --it;
      --it;
      --it;
      goto_program.insert_swap(it++, *if_statement);

      /* reset
        After insertion, reset the pointer `it`'s position to the `origin_1`
      */
      ++it;
      ++it;
      ++it;
      goto_program.compute_target_numbers();
    }
    else
    {
      // rhs
      exprt rhs = exprt("sideeffect", l_t);
      rhs.statement("nondet");
      rhs.location() = l;

      // assignment
      goto_programt tmp;
      goto_programt::targett assignment = tmp.add_instruction(ASSIGN);
      assignment->location = l;
      assignment->function = it->location.get_function();

      code_assignt code_assign(lhs, rhs);
      code_assign.location() = it->location;
      expr2tc new_assign_expr;
      migrate_expr(code_assign, new_assign_expr);
      assignment->code = new_assign_expr;

      // insert
      goto_program.insert_swap(it++, *assignment);
      --it;
    }
  }

  goto_program.update();
  return true;
}