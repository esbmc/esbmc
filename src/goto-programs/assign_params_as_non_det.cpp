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

bool assign_params_as_non_det::assign_nondet(
  const exprt &arg,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it,
  locationt l)
{
  // lhs
  const auto &_id = arg.get("#identifier").empty() ? arg.get("identifier")
                                                   : arg.get("#identifier");
  if (context.find_symbol(_id) == nullptr)
  {
    log_error("Unepected");
    return true; // Not expected
  }
  exprt lhs = symbol_expr(*context.find_symbol(_id));
  typet l_t = lhs.type();

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

  return false;
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

  if (F.first.as_string() == "c:@F@main" && t.arguments().size() == 2)
  {
    /*
    argc = NONDET;
    assume(argc >= 1);
    assume(argc <= 268435454);
    char **argv = malloc(argc * sizeof(char *));
    */
    namespacet ns(context);

    const symbolt &argc_symbol = *ns.lookup("argc'");
    const symbolt &argv_symbol = *ns.lookup("argv'");
    exprt argc =
      symbol_exprt(t.arguments().at(0).get("#identifier"), argc_symbol.type);
    exprt argv =
      symbol_exprt(t.arguments().at(1).get("#identifier"), argv_symbol.type);

    // argc = NONDET;
    if (assign_nondet(argc, goto_program, it, l))
      return true;

    // assume argc is at least one. assume(argc >= 1);
    exprt one = from_integer(1, argc_symbol.type);
    exprt ge(">=", bool_type());
    ge.copy_to_operands(argc, one);

    // do assume
    expr2tc n_guard;
    migrate_expr(ge, n_guard);
    goto_programt::instructiont instruction;
    instruction.make_assumption(n_guard);
    instruction.location = l;
    instruction.location.property(
      "skipped"); // we do not calculate this condition/branch
    instruction.function = it->location.get_function();
    goto_program.insert_swap(it++, instruction);
    --it;

    // assume argc is at most MAX-1
    BigInt max;

    if (argc_symbol.type.id() == "signedbv")
      max = power(2, atoi(argc_symbol.type.width().c_str()) - 1) - 1;
    else if (argc_symbol.type.id() == "unsignedbv")
      max = power(2, atoi(argc_symbol.type.width().c_str())) - 1;
    else
      assert(false);

    // The argv array of MAX elements of pointer type has to fit
    max /= config.ansi_c.pointer_width() / 8;

    exprt max_minus_one = from_integer(max - 1, argc_symbol.type);

    exprt le("<=", bool_type());
    le.copy_to_operands(argc, max_minus_one);

    expr2tc n_guard2;
    migrate_expr(le, n_guard2);
    goto_programt::instructiont instruction2;
    instruction2.make_assumption(n_guard2);
    instruction2.location = l;
    instruction2.location.property(
      "skipped"); // we do not calculate this condition/branch
    instruction2.function = it->location.get_function();
    goto_program.insert_swap(it++, instruction2);
    --it;

    // argv = malloc(argc * sizeof(char *));

    side_effect_expr_function_callt call_expr;
    call_expr.function() =
      symbol_expr(*context.find_symbol("c:@F@ESBMC_malloc_argv"));
    call_expr.arguments().push_back(argc);
    call_expr.arguments().push_back(argv);

    code_function_callt function_call;
    function_call.lhs() = argv;
    function_call.function() = call_expr.function();
    function_call.arguments() = call_expr.arguments();
    function_call.location() = l;

    // do assignment
    goto_programt dest;
    goto_programt::targett t = dest.add_instruction(FUNCTION_CALL);
    migrate_expr(function_call, t->code);
    t->location = function_call.location();

    goto_programt::instructiont instruction3 = dest.instructions.front();
    goto_program.insert_swap(it++, instruction3);
    --it;
  }
  else
  {
    for (const auto &arg : t.arguments())
    {
      if (assign_nondet(arg, goto_program, it, l))
        return true;
      exprt lhs = symbol_expr(*context.find_symbol(arg.get("#identifier")));
      typet l_t = lhs.type();

      // additional handling for non-void pointer
      // so the pointer can point to an array (decay), which is covered above
      // also, the pointer can point to an object/variable, which is covered below
      if (l_t.is_pointer() && l_t.subtype() != empty_typet())
      {
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
        
        During this process we create two auxiliary variable. One for the boolean flag, and one for the object.
        - We need to consider the situation where pointers can be null, thus we put the assignment under an if-statement
        - If the flag (`lhs#temp_bool`) is true, we create an object (`lhs#temp`) and assign its address to the pointer.
      */

        // lhs = null;
        exprt zero_rhs = exprt("sideeffect", l_t);
        zero_rhs.statement("nondet");
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
        goto_programt::targett bool_assignment =
          bool_tmp.add_instruction(ASSIGN);
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
        instruction.location.property(
          "skipped"); // we do not calculate this condition/branch
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
        expr2tc not_guard;
        migrate_expr(symbol_expr(*new_sym2), not_guard);
        // make not: !lhs#temp_bool
        make_not(not_guard);

        // inser if statement to the goto program
        goto_programt tmp4;
        goto_programt::targett if_statement = tmp4.add_instruction();
        if_statement->location = l;
        if_statement->location.property(
          "skipped"); // we do not calculate this condition/branch
        if_statement->function = it->location.get_function();

        if_statement->make_goto(it);
        if_statement->guard = not_guard;

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
    }
  }
  goto_program.update();
  return false;
}