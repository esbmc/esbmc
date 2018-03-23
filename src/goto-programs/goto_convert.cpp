/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com
		Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <cassert>
#include <goto-programs/destructor.h>
#include <goto-programs/goto_convert_class.h>
#include <goto-programs/remove_skip.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/i2string.h>
#include <util/irep2_utils.h>
#include <util/prefix.h>
#include <util/std_expr.h>
#include <util/type_byte_size.h>

//#define DEBUG

#ifdef DEBUG
#define DEBUGLOC                                                               \
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#else
#define DEBUGLOC
#endif

void goto_convertt::finish_gotos()
{
  for(auto it : targets.gotos)
  {
    goto_programt::instructiont &i = *it;

    if(is_code_goto2t(i.code))
    {
      const irep_idt &goto_label = to_code_goto2t(i.code).target;

      labelst::const_iterator l_it = targets.labels.find(goto_label);

      if(l_it == targets.labels.end())
      {
        std::cerr << "goto label " << goto_label << " not found";
        i.code->dump();
        abort();
      }

      i.targets.clear();
      i.targets.push_back(l_it->second);
    }
    else
    {
      err_location(migrate_expr_back(i.code));
      throw "finish_gotos: unexpected goto";
    }
  }

  targets.gotos.clear();
}

void goto_convertt::goto_convert(const codet &code, goto_programt &dest)
{
  goto_convert_rec(code, dest);
}

void goto_convertt::goto_convert_rec(const codet &code, goto_programt &dest)
{
  convert(code, dest);

  finish_gotos();
}

void goto_convertt::copy(
  const codet &code,
  goto_program_instruction_typet type,
  goto_programt &dest)
{
  goto_programt::targett t = dest.add_instruction(type);
  migrate_expr(code, t->code);
  t->location = code.location();
}

void goto_convertt::convert_label(const code_labelt &code, goto_programt &dest)
{
  if(code.operands().size() != 1)
  {
    err_location(code);
    throw "label statement expected to have one operand";
  }

  // grab the label
  const irep_idt &label = code.get_label();

  goto_programt tmp;

  convert(to_code(code.op0()), tmp);

  // magic ERROR label?

  const std::string &error_label = options.get_option("error-label");

  goto_programt::targett target;

  if(error_label != "" && label == error_label)
  {
    goto_programt::targett t = dest.add_instruction(ASSERT);
    t->guard = gen_false_expr();
    t->location = code.location();
    t->location.property("error label");
    t->location.comment("error label");
    t->location.user_provided(false);

    target = t;
    dest.destructive_append(tmp);
  }
  else
  {
    target = tmp.instructions.begin();
    dest.destructive_append(tmp);
  }

  if(!label.empty())
  {
    targets.labels.insert(
      std::pair<irep_idt, goto_programt::targett>(label, target));
    target->labels.push_back(label);
  }
}

void goto_convertt::convert_switch_case(
  const code_switch_caset &code,
  goto_programt &dest)
{
  if(code.operands().size() != 2)
  {
    err_location(code);
    throw "switch-case statement expected to have two operands";
  }

  goto_programt tmp;
  convert(code.code(), tmp);

  goto_programt::targett target = tmp.instructions.begin();
  dest.destructive_append(tmp);

  // default?

  if(code.is_default())
    targets.set_default(target);
  else
  {
    // cases?

    const exprt &case_op = code.case_op();
    targets.cases[target].push_back(case_op);
  }
}

void goto_convertt::convert(const codet &code, goto_programt &dest)
{
  const irep_idt &statement = code.get_statement();

  if(statement == "block")
    convert_block(code, dest);
  else if(statement == "decl")
    convert_decl(code, dest);
  else if(statement == "decl-block")
    convert_decl_block(code, dest);
  else if(statement == "expression")
    convert_expression(code, dest);
  else if(statement == "assign")
    convert_assign(to_code_assign(code), dest);
  else if(statement == "init")
    convert_init(code, dest);
  else if(statement == "assert")
    convert_assert(code, dest);
  else if(statement == "assume")
    convert_assume(code, dest);
  else if(statement == "function_call")
    convert_function_call(to_code_function_call(code), dest);
  else if(statement == "label")
    convert_label(to_code_label(code), dest);
  else if(statement == "switch_case")
    convert_switch_case(to_code_switch_case(code), dest);
  else if(statement == "for")
    convert_for(code, dest);
  else if(statement == "while")
    convert_while(code, dest);
  else if(statement == "dowhile")
    convert_dowhile(code, dest);
  else if(statement == "switch")
    convert_switch(code, dest);
  else if(statement == "break")
    convert_break(to_code_break(code), dest);
  else if(statement == "return")
    convert_return(to_code_return(code), dest);
  else if(statement == "continue")
    convert_continue(to_code_continue(code), dest);
  else if(statement == "goto")
    convert_goto(code, dest);
  else if(statement == "skip")
    convert_skip(code, dest);
  else if(statement == "non-deterministic-goto")
    convert_non_deterministic_goto(code, dest);
  else if(statement == "ifthenelse")
    convert_ifthenelse(code, dest);
  else if(statement == "atomic_begin")
    convert_atomic_begin(code, dest);
  else if(statement == "atomic_end")
    convert_atomic_end(code, dest);
  else if(statement == "cpp_delete" || statement == "cpp_delete[]")
    convert_cpp_delete(code, dest);
  else if(statement == "cpp-catch")
    convert_catch(code, dest);
  else if(statement == "throw_decl")
    convert_throw_decl(code, dest);
  else if(statement == "throw_decl_end")
    convert_throw_decl_end(code, dest);
  else
  {
    copy(code, OTHER, dest);
  }

  // if there is no instruction in the program, add skip to it
  if(dest.instructions.empty())
  {
    dest.add_instruction(SKIP);
    dest.instructions.back().code = expr2tc();
  }
}

void goto_convertt::convert_throw_decl_end(
  const exprt &expr,
  goto_programt &dest)
{
  // add the THROW_DECL_END instruction to 'dest'
  goto_programt::targett throw_decl_end_instruction = dest.add_instruction();
  throw_decl_end_instruction->make_throw_decl_end();
  throw_decl_end_instruction->code = code_cpp_throw_decl_end2tc();
  throw_decl_end_instruction->location = expr.location();
}

void goto_convertt::convert_throw_decl(const exprt &expr, goto_programt &dest)
{
  // add the THROW_DECL instruction to 'dest'
  goto_programt::targett throw_decl_instruction = dest.add_instruction();
  codet c("code");
  c.set_statement("throw-decl");

  // the THROW_DECL instruction is annotated with a list of IDs,
  // one per target
  irept::subt &throw_list = c.add("throw_list").get_sub();
  for(const auto &block : expr.operands())
  {
    irept type = irept(block.get("throw_decl_id"));

    // grab the ID and add to THROW_DECL instruction
    throw_list.emplace_back(type);
  }

  throw_decl_instruction->make_throw_decl();
  throw_decl_instruction->location = expr.location();
  migrate_expr(c, throw_decl_instruction->code);
}

void goto_convertt::convert_catch(const codet &code, goto_programt &dest)
{
  assert(code.operands().size() >= 2);

  // add the CATCH-push instruction to 'dest'
  goto_programt::targett catch_push_instruction = dest.add_instruction();
  catch_push_instruction->make_catch();
  catch_push_instruction->location = code.location();

  // the CATCH-push instruction is annotated with a list of IDs,
  // one per target.
  std::vector<irep_idt> exception_list;

  // add a SKIP target for the end of everything
  goto_programt end;
  goto_programt::targett end_target = end.add_instruction();
  end_target->make_skip();

  // the first operand is the 'try' block
  goto_programt tmp;
  convert(to_code(code.op0()), tmp);
  dest.destructive_append(tmp);

  // add the CATCH-pop to the end of the 'try' block
  goto_programt::targett catch_pop_instruction = dest.add_instruction();
  catch_pop_instruction->make_catch();
  std::vector<irep_idt> empty_excp_list;
  catch_pop_instruction->code = code_cpp_catch2tc(empty_excp_list);

  // add a goto to the end of the 'try' block
  dest.add_instruction()->make_goto(end_target);

  for(unsigned i = 1; i < code.operands().size(); i++)
  {
    const codet &block = to_code(code.operands()[i]);

    // grab the ID and add to CATCH instruction
    exception_list.push_back(block.get("exception_id"));

    // Hack for object value passing
    const_cast<exprt::operandst &>(block.op0().operands())
      .push_back(gen_zero(block.op0().op0().type()));

    convert(block, tmp);
    catch_push_instruction->targets.push_back(tmp.instructions.begin());
    dest.destructive_append(tmp);

    // add a goto to the end of the 'catch' block
    dest.add_instruction()->make_goto(end_target);
  }

  // add end-target
  dest.destructive_append(end);

  catch_push_instruction->code = code_cpp_catch2tc(exception_list);
}

void goto_convertt::convert_block(const codet &code, goto_programt &dest)
{
  // Save local symbols
  goto_programt::local_variablest old_scoped_vars = scoped_variables;
  scoped_variables.clear();

  // Convert each expression
  for(auto const &it : code.operands())
  {
    const codet &code_it = to_code(it);

    goto_programt tmp;
    convert(code_it, tmp);

    dest.destructive_append(tmp);
  }

  // see if we need to call any destructors
  for(auto const &local : scoped_variables)
  {
    const symbolt &symbol = ns.lookup(local);

    code_function_callt destructor = get_destructor(ns, symbol.type);
    if(destructor.is_not_nil())
    {
      // add "this"
      exprt this_expr("address_of", pointer_typet());
      this_expr.type().subtype() = symbol.type;
      this_expr.copy_to_operands(symbol_expr(symbol));
      destructor.arguments().push_back(this_expr);

      goto_programt tmp;
      convert(destructor, tmp);

      dest.destructive_append(tmp);
    }
  }

  // Add scoped variables to the list of function variables
  local_variables.insert(
    local_variables.begin(), scoped_variables.begin(), scoped_variables.end());

  // Add old symbols to the list of locals
  scoped_variables = old_scoped_vars;
}

void goto_convertt::convert_expression(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 1)
  {
    err_location(code);
    throw "expression statement takes one operand";
  }

  exprt expr = code.op0();

  if(expr.id() == "if")
  {
    const if_exprt &if_expr = to_if_expr(expr);
    code_ifthenelset tmp_code;
    tmp_code.location() = expr.location();
    tmp_code.cond() = if_expr.cond();
    tmp_code.then_case() = code_expressiont(if_expr.true_case());
    tmp_code.then_case().location() = expr.location();
    tmp_code.else_case() = code_expressiont(if_expr.false_case());
    tmp_code.else_case().location() = expr.location();
    convert_ifthenelse(tmp_code, dest);
  }
  else
  {
    remove_sideeffects(expr, dest, false); // result not used

    if(expr.is_not_nil())
    {
      codet tmp(code);
      tmp.op0() = expr;
      tmp.location() = expr.location();
      copy(tmp, OTHER, dest);
    }
  }
}

bool goto_convertt::rewrite_vla_decl_size(exprt &size, goto_programt &dest)
{
  // Remove side effect
  if(has_sideeffect(size))
  {
    goto_programt sideeffects;
    remove_sideeffects(size, sideeffects);
    dest.destructive_append(sideeffects);
    return true;
  }

  // We have to replace the symbol by a temporary, because it might
  // change its value in the future
  // Don't create a symbol for temporary symbols
  if(
    size.is_symbol() &&
    size.identifier().as_string().find("tmp$") == std::string::npos)
  {
    // Old size symbol
    exprt old_size = size;

    // Replace the size by a new variable, to avoid wrong results
    // when the symbol used to create the VLA is changed
    size = symbol_expr(new_tmp_symbol(size.type()));

    codet assignment("assign");
    assignment.reserve_operands(2);
    assignment.copy_to_operands(size);
    assignment.copy_to_operands(old_size);
    assignment.location() = old_size.location();
    copy(assignment, ASSIGN, dest);

    return true;
  }

  // A constant array
  return false;
}

bool goto_convertt::rewrite_vla_decl(typet &var_type, goto_programt &dest)
{
  // Not an array, don't care
  if(!var_type.is_array())
    return false;

  array_typet &arr_type = to_array_type(var_type);

  // Rewrite size
  bool res = rewrite_vla_decl_size(arr_type.size(), dest);

  // It's a multidimensional array, apply the transformations recursively.
  // res is the second operand because it can be short-circuited and the
  // side-effect will not be evaluated
  if(arr_type.subtype().is_array())
    return rewrite_vla_decl(to_array_type(arr_type.subtype()), dest) || res;

  // Now rewrite the size expression
  return res;
}

void goto_convertt::generate_dynamic_size_vla(
  exprt &var,
  const locationt &loc,
  goto_programt &dest)
{
  assert(var.type().is_array());

  array_typet arr_type = to_array_type(var.type());
  exprt size = to_array_type(var.type()).size();

  // First, if it's a multidimensional vla, the size will be the
  // multiplication of the dimensions
  while(arr_type.subtype().is_array())
  {
    array_typet arr_subtype = to_array_type(arr_type.subtype());

    exprt mult(exprt::mult, size.type());
    mult.copy_to_operands(size, arr_subtype.size());
    size.swap(mult);

    arr_type = arr_subtype;
  }

  // Now, calculate the array size, which are the dimensions times the
  // elements' size
  const typet &subtype = arr_type.subtype();

  type2tc tmp;
  migrate_type(subtype, tmp);
  auto st_size = type_byte_size(tmp);

  exprt st_size_expr = from_integer(st_size, size.type());
  exprt mult(exprt::mult, size.type());
  mult.copy_to_operands(size, st_size_expr);

  // Set the array to have a dynamic size
  address_of_exprt addrof(var);
  exprt dynamic_size("dynamic_size", int_type());
  dynamic_size.copy_to_operands(addrof);

  goto_programt::targett t_s_s = dest.add_instruction(ASSIGN);
  exprt assign = code_assignt(dynamic_size, mult);
  migrate_expr(assign, t_s_s->code);
  t_s_s->location = loc;
}

void goto_convertt::convert_decl(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 1 && code.operands().size() != 2)
  {
    err_location(code);
    throw "decl statement takes one or two operands";
  }

  // We might change the symbol
  codet new_code(code);

  exprt &var = new_code.op0();
  if(!var.is_symbol())
  {
    err_location(var);
    throw "decl statement expects symbol as first operand";
  }

  const irep_idt &identifier = var.identifier();

  symbolt *s = context.find_symbol(identifier);
  assert(s != nullptr);

  // A static variable will be declared in the global scope and
  // a code type means a function declaration, we ignore both
  if(s->static_lifetime || s->type.is_code())
    return; // this is a SKIP!

  // Local variable, add to locals
  scoped_variables.push_front(identifier);

  // Check if is an VLA declaration and rewrite the declaration
  bool is_vla = rewrite_vla_decl(var.type(), dest);
  if(is_vla)
  {
    // This means that it was a VLA declaration and we need to
    // to rewrite the symbol as well
    s->type = var.type();
  }

  exprt initializer = nil_exprt();
  if(new_code.operands().size() == 2)
  {
    initializer = new_code.op1();

    // just resize the vector, this will get rid of op1
    new_code.operands().pop_back();

    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(initializer);
      if(globals > 0)
        break_globals2assignments(initializer, dest, new_code.location());
    }

    goto_programt sideeffects;
    remove_sideeffects(initializer, sideeffects);
    dest.destructive_append(sideeffects);
  }

  // break up into decl and assignment
  copy(new_code, OTHER, dest);

  if(is_vla)
    generate_dynamic_size_vla(var, new_code.location(), dest);

  if(!initializer.is_nil())
  {
    code_assignt assign(var, initializer);
    assign.location() = new_code.location();
    copy(assign, ASSIGN, dest);
  }
}

void goto_convertt::convert_decl_block(const codet &code, goto_programt &dest)
{
  for(auto const &it : code.operands())
    convert(to_code(it), dest);
}

void goto_convertt::convert_assign(
  const code_assignt &code,
  goto_programt &dest)
{
  if(code.operands().size() != 2)
  {
    err_location(code);
    throw "assignment statement takes two operands";
  }

  exprt lhs = code.lhs(), rhs = code.rhs();

  remove_sideeffects(lhs, dest);

  if(rhs.id() == "sideeffect" && rhs.statement() == "function_call")
  {
    if(rhs.operands().size() != 2)
    {
      err_location(rhs);
      throw "function_call sideeffect takes two operands";
    }

    Forall_operands(it, rhs)
      remove_sideeffects(*it, dest);

    do_function_call(lhs, rhs.op0(), rhs.op1().operands(), dest);
  }
  else if(
    rhs.id() == "sideeffect" &&
    (rhs.statement() == "cpp_new" || rhs.statement() == "cpp_new[]"))
  {
    Forall_operands(it, rhs)
      remove_sideeffects(*it, dest);

    do_cpp_new(lhs, rhs, dest);
  }
  else
  {
    remove_sideeffects(rhs, dest);

    if(rhs.type().is_code())
    {
      convert(to_code(rhs), dest);
      return;
    }

    if(lhs.id() == "typecast")
    {
      assert(lhs.operands().size() == 1);

      // move to rhs
      exprt tmp_rhs(lhs);
      tmp_rhs.op0() = rhs;
      rhs = tmp_rhs;

      // remove from lhs
      exprt tmp(lhs.op0());
      lhs.swap(tmp);
    }

    int atomic = 0;
    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(lhs);
      atomic = globals;
      globals += get_expr_number_globals(rhs);
      if(
        globals > 0 &&
        (lhs.identifier().as_string().find("tmp$") == std::string::npos))
        break_globals2assignments(atomic, lhs, rhs, dest, code.location());
    }

    code_assignt new_assign(code);
    new_assign.lhs() = lhs;
    new_assign.rhs() = rhs;
    copy(new_assign, ASSIGN, dest);

    if(options.get_bool_option("atomicity-check"))
      if(atomic == -1)
        dest.add_instruction(ATOMIC_END);
  }
}

void goto_convertt::break_globals2assignments(
  int &atomic,
  exprt &lhs,
  exprt &rhs,
  goto_programt &dest,
  const locationt &location)
{
  if(!options.get_bool_option("atomicity-check"))
    return;

  exprt atomic_dest = exprt("and", typet("bool"));

  /* break statements such as a = b + c as follows:
   * tmp1 = b;
   * tmp2 = c;
   * atomic_begin
   * assert tmp1==b && tmp2==c
   * a = b + c
   * atomic_end
  */
  //break_globals2assignments_rec(lhs,atomic_dest,dest,atomic,location);
  break_globals2assignments_rec(rhs, atomic_dest, dest, atomic, location);

  if(atomic_dest.operands().size() == 1)
  {
    exprt tmp;
    tmp.swap(atomic_dest.op0());
    atomic_dest.swap(tmp);
  }
  if(atomic_dest.operands().size() != 0)
  {
    // do an assert
    if(atomic > 0)
    {
      dest.add_instruction(ATOMIC_BEGIN);
      atomic = -1;
    }
    goto_programt::targett t = dest.add_instruction(ASSERT);
    expr2tc tmp_guard;
    migrate_expr(atomic_dest, tmp_guard);
    t->guard = tmp_guard;
    t->location = location;
    t->location.comment(
      "atomicity violation on assignment to " + lhs.identifier().as_string());
  }
}

void goto_convertt::break_globals2assignments(
  exprt &rhs,
  goto_programt &dest,
  const locationt &location)
{
  if(!options.get_bool_option("atomicity-check"))
    return;

  if(rhs.operands().size() > 0)
    if(rhs.op0().identifier().as_string().find("pthread") != std::string::npos)
      return;

  if(rhs.operands().size() > 0)
    if(rhs.op0().operands().size() > 0)
      return;

  exprt atomic_dest = exprt("and", typet("bool"));
  break_globals2assignments_rec(rhs, atomic_dest, dest, 0, location);

  if(atomic_dest.operands().size() == 1)
  {
    exprt tmp;
    tmp.swap(atomic_dest.op0());
    atomic_dest.swap(tmp);
  }

  if(atomic_dest.operands().size() != 0)
  {
    goto_programt::targett t = dest.add_instruction(ASSERT);
    expr2tc tmp_dest;
    migrate_expr(atomic_dest, tmp_dest);
    t->guard.swap(tmp_dest);
    t->location = location;
    t->location.comment("atomicity violation");
  }
}

void goto_convertt::break_globals2assignments_rec(
  exprt &rhs,
  exprt &atomic_dest,
  goto_programt &dest,
  int atomic,
  const locationt &location)
{
  if(!options.get_bool_option("atomicity-check"))
    return;

  if(
    rhs.id() == "dereference" || rhs.id() == "implicit_dereference" ||
    rhs.id() == "index" || rhs.id() == "member")
  {
    irep_idt identifier = rhs.op0().identifier();
    if(rhs.id() == "member")
    {
      const exprt &object = rhs.operands()[0];
      identifier = object.identifier();
    }
    else if(rhs.id() == "index")
    {
      identifier = rhs.op1().identifier();
    }

    if(identifier.empty())
      return;

    const symbolt &symbol = ns.lookup(identifier);

    if(
      !(identifier == "__ESBMC_alloc" || identifier == "__ESBMC_alloc_size") &&
      (symbol.static_lifetime || symbol.type.is_dynamic_set()))
    {
      // make new assignment to temp for each global symbol
      symbolt &new_symbol = new_tmp_symbol(rhs.type());
      equality_exprt eq_expr;
      irept irep;
      new_symbol.to_irep(irep);
      eq_expr.lhs() = symbol_expr(new_symbol);
      eq_expr.rhs() = rhs;
      atomic_dest.copy_to_operands(eq_expr);

      codet assignment("assign");
      assignment.reserve_operands(2);
      assignment.copy_to_operands(symbol_expr(new_symbol));
      assignment.copy_to_operands(rhs);
      assignment.location() = location;
      assignment.comment("atomicity violation");
      copy(assignment, ASSIGN, dest);

      if(atomic == 0)
        rhs = symbol_expr(new_symbol);
    }
  }
  else if(rhs.id() == "symbol")
  {
    const irep_idt &identifier = rhs.identifier();
    const symbolt &symbol = ns.lookup(identifier);
    if(symbol.static_lifetime || symbol.type.is_dynamic_set())
    {
      // make new assignment to temp for each global symbol
      symbolt &new_symbol = new_tmp_symbol(rhs.type());
      new_symbol.static_lifetime = true;
      equality_exprt eq_expr;
      irept irep;
      new_symbol.to_irep(irep);
      eq_expr.lhs() = symbol_expr(new_symbol);
      eq_expr.rhs() = rhs;
      atomic_dest.copy_to_operands(eq_expr);

      codet assignment("assign");
      assignment.reserve_operands(2);
      assignment.copy_to_operands(symbol_expr(new_symbol));
      assignment.copy_to_operands(rhs);

      assignment.location() = rhs.find_location();
      assignment.comment("atomicity violation");
      copy(assignment, ASSIGN, dest);

      if(atomic == 0)
        rhs = symbol_expr(new_symbol);
    }
  }
  else if(!rhs.is_address_of()) // && rhs.id() != "dereference")
  {
    Forall_operands(it, rhs)
    {
      break_globals2assignments_rec(*it, atomic_dest, dest, atomic, location);
    }
  }
}

unsigned int goto_convertt::get_expr_number_globals(const exprt &expr)
{
  if(!options.get_bool_option("atomicity-check"))
    return 0;

  if(expr.is_address_of())
    return 0;

  if(expr.id() == "symbol")
  {
    const irep_idt &identifier = expr.identifier();
    const symbolt &symbol = ns.lookup(identifier);

    if(identifier == "__ESBMC_alloc" || identifier == "__ESBMC_alloc_size")
    {
      return 0;
    }
    if(symbol.static_lifetime || symbol.type.is_dynamic_set())
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }

  unsigned int globals = 0;

  forall_operands(it, expr)
    globals += get_expr_number_globals(*it);

  return globals;
}

unsigned int goto_convertt::get_expr_number_globals(const expr2tc &expr)
{
  if(is_nil_expr(expr))
    return 0;

  if(!options.get_bool_option("atomicity-check"))
    return 0;

  if(is_address_of2t(expr))
    return 0;
  if(is_symbol2t(expr))
  {
    irep_idt identifier = to_symbol2t(expr).get_symbol_name();
    const symbolt &symbol = ns.lookup(identifier);

    if(identifier == "__ESBMC_alloc" || identifier == "__ESBMC_alloc_size")
    {
      return 0;
    }
    if(symbol.static_lifetime || symbol.type.is_dynamic_set())
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }

  unsigned int globals = 0;

  expr->foreach_operand([this, &globals](const expr2tc &e) {
    globals += get_expr_number_globals(e);
  });

  return globals;
}

void goto_convertt::convert_init(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 2)
  {
    err_location(code);
    throw "init statement takes two operands";
  }

  // make it an assignment
  codet assignment = code;
  assignment.set_statement("assign");

  convert(to_code_assign(assignment), dest);
}

void goto_convertt::convert_cpp_delete(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 1)
  {
    err_location(code);
    throw "cpp_delete statement takes one operand";
  }

  exprt tmp_op = code.op0();

  // we call the destructor, and then free
  const exprt &destructor = static_cast<const exprt &>(code.find("destructor"));

  if(destructor.is_not_nil())
  {
    if(code.statement() == "cpp_delete[]")
    {
      // build loop
    }
    else if(code.statement() == "cpp_delete")
    {
      exprt deref_op("dereference", tmp_op.type().subtype());
      deref_op.copy_to_operands(tmp_op);

      codet tmp_code = to_code(destructor);
      replace_new_object(deref_op, tmp_code);
      convert(tmp_code, dest);
    }
    else
      assert(0);
  }

  expr2tc tmp_op2;
  migrate_expr(tmp_op, tmp_op2);

  // preserve the call
  goto_programt::targett t_f = dest.add_instruction(OTHER);
  t_f->code = code_cpp_delete2tc(tmp_op2);
  t_f->location = code.location();

  // now do "delete"
  exprt valid_expr("valid_object", bool_typet());
  valid_expr.copy_to_operands(tmp_op);

  // clear alloc bit
  exprt assign = code_assignt(valid_expr, false_exprt());
  expr2tc assign2;
  migrate_expr(assign, assign2);
  goto_programt::targett t_c = dest.add_instruction(ASSIGN);
  t_c->code = assign2;
  t_c->location = code.location();

  exprt deallocated_expr("deallocated_object", bool_typet());
  deallocated_expr.copy_to_operands(tmp_op);

  //indicate that memory has been deallocated
  assign = code_assignt(deallocated_expr, true_exprt());
  migrate_expr(assign, assign2);
  goto_programt::targett t_d = dest.add_instruction(ASSIGN);
  t_d->code = assign2;
  t_d->location = code.location();
}

void goto_convertt::convert_assert(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 1)
  {
    err_location(code);
    throw "assert statement takes one operand";
  }

  exprt cond = code.op0();

  remove_sideeffects(cond, dest);

  if(options.get_bool_option("no-assertions"))
    return;

  if(options.get_bool_option("atomicity-check"))
  {
    unsigned int globals = get_expr_number_globals(cond);
    if(globals > 0)
      break_globals2assignments(cond, dest, code.location());
  }

  goto_programt::targett t = dest.add_instruction(ASSERT);
  expr2tc tmp_cond;
  migrate_expr(cond, tmp_cond);
  t->guard = tmp_cond;
  t->location = code.location();
  t->location.property("assertion");
  t->location.user_provided(true);
}

void goto_convertt::convert_skip(const codet &code, goto_programt &dest)
{
  goto_programt::targett t = dest.add_instruction(SKIP);
  t->location = code.location();
  expr2tc tmp_code;
  migrate_expr(code, tmp_code);
  t->code = tmp_code;
}

void goto_convertt::convert_assume(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 1)
  {
    err_location(code);
    throw "assume statement takes one operand";
  }

  exprt op = code.op0();

  remove_sideeffects(op, dest);

  if(options.get_bool_option("atomicity-check"))
  {
    unsigned int globals = get_expr_number_globals(op);
    if(globals > 0)
      break_globals2assignments(op, dest, code.location());
  }

  goto_programt::targett t = dest.add_instruction(ASSUME);
  expr2tc tmp_op;
  migrate_expr(op, tmp_op);
  t->guard.swap(tmp_op);
  t->location = code.location();
}

void goto_convertt::convert_for(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 4)
  {
    err_location(code);
    throw "for takes four operands";
  }

  // turn for(A; c; B) { P } into
  //  A; while(c) { P; B; }
  //-----------------------------
  //    A;
  // u: sideeffects in c
  // v: if(!c) goto z;
  // w: P;
  // x: B;               <-- continue target
  // y: goto u;
  // z: ;                <-- break target

  // A;
  code_blockt block;
  if(code.op0().is_not_nil())
  {
    block.copy_to_operands(code.op0());
    convert(block, dest);
  }

  exprt tmp = code.op1();

  exprt cond = tmp;
  goto_programt sideeffects;

  remove_sideeffects(cond, sideeffects);

  // save break/continue targets
  break_continue_targetst old_targets(targets);

  // do the u label
  goto_programt::targett u = sideeffects.instructions.begin();

  // do the v label
  goto_programt tmp_v;
  goto_programt::targett v = tmp_v.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z = tmp_z.add_instruction(SKIP);
  z->location = code.location();

  // do the x label
  goto_programt tmp_x;
  if(code.op2().is_nil())
  {
    tmp_x.add_instruction(SKIP);
    tmp_x.instructions.back().location = code.location();
  }
  else
  {
    exprt tmp_B = code.op2();
    convert(to_code(code.op2()), tmp_x);
  }

  // optimize the v label
  if(sideeffects.instructions.empty())
    u = v;

  // set the targets
  targets.set_break(z);
  targets.set_continue(tmp_x.instructions.begin());

  // v: if(!c) goto z;
  v->make_goto(z);
  expr2tc tmp_cond;
  migrate_expr(cond, tmp_cond);
  tmp_cond = not2tc(tmp_cond);
  v->guard = tmp_cond;
  v->location = cond.location();

  // do the w label
  goto_programt tmp_w;
  convert(to_code(code.op3()), tmp_w);

  // y: goto u;
  goto_programt tmp_y;
  goto_programt::targett y = tmp_y.add_instruction();
  y->make_goto(u);
  y->guard = gen_true_expr();
  y->location = code.location();

  dest.destructive_append(sideeffects);
  dest.destructive_append(tmp_v);
  dest.destructive_append(tmp_w);
  dest.destructive_append(tmp_x);
  dest.destructive_append(tmp_y);
  dest.destructive_append(tmp_z);

  // restore break/continue
  targets.restore(old_targets);
}

void goto_convertt::convert_while(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 2)
  {
    err_location(code);
    throw "while takes two operands";
  }

  exprt tmp = code.op0();
  const exprt *cond = &tmp;
  const locationt &location = code.location();

  //    while(c) P;
  //--------------------
  // v: if(!c) goto z;
  // x: P;
  // y: goto v;          <-- continue target
  // z: ;                <-- break target

  // save break/continue targets
  break_continue_targetst old_targets(targets);

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z = tmp_z.add_instruction();
  z->make_skip();
  z->location = location;

  goto_programt tmp_branch;
  generate_conditional_branch(gen_not(*cond), z, location, tmp_branch);

  // do the v label
  goto_programt::targett v = tmp_branch.instructions.begin();

  // do the y label
  goto_programt tmp_y;
  goto_programt::targett y = tmp_y.add_instruction();

  // set the targets
  targets.set_break(z);
  targets.set_continue(y);

  // do the x label
  goto_programt tmp_x;
  convert(to_code(code.op1()), tmp_x);

  // y: if(c) goto v;
  y->make_goto(v);
  y->guard = gen_true_expr();
  y->location = code.location();

  dest.destructive_append(tmp_branch);
  dest.destructive_append(tmp_x);
  dest.destructive_append(tmp_y);
  dest.destructive_append(tmp_z);

  // restore break/continue
  targets.restore(old_targets);
}

void goto_convertt::convert_dowhile(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 2)
  {
    err_location(code);
    throw "dowhile takes two operands";
  }

  // save location
  locationt condition_location = code.op0().find_location();

  exprt tmp = code.op0();

  goto_programt sideeffects;
  remove_sideeffects(tmp, sideeffects);
  const exprt &cond = tmp;

  //    do P while(c);
  //--------------------
  // w: P;
  // x: sideeffects in c   <-- continue target
  // y: if(c) goto w;
  // z: ;                  <-- break target

  // save break/continue targets
  break_continue_targetst old_targets(targets);

  // do the y label
  goto_programt tmp_y;
  goto_programt::targett y = tmp_y.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z = tmp_z.add_instruction();
  z->make_skip();
  z->location = code.location();

  // do the x label
  goto_programt::targett x;
  if(sideeffects.instructions.empty())
    x = y;
  else
    x = sideeffects.instructions.begin();

  // set the targets
  targets.set_break(z);
  targets.set_continue(x);

  // do the w label
  goto_programt tmp_w;
  convert(to_code(code.op1()), tmp_w);
  goto_programt::targett w = tmp_w.instructions.begin();

  // y: if(c) goto w;
  y->make_goto(w);
  migrate_expr(cond, y->guard);
  y->location = condition_location;

  dest.destructive_append(tmp_w);
  dest.destructive_append(sideeffects);
  dest.destructive_append(tmp_y);
  dest.destructive_append(tmp_z);

  // restore break/continue targets
  targets.restore(old_targets);
}

void goto_convertt::case_guard(
  const exprt &value,
  const exprt::operandst &case_op,
  exprt &dest)
{
  dest = exprt("or", typet("bool"));
  dest.reserve_operands(case_op.size());

  forall_expr(it, case_op)
  {
    equality_exprt eq_expr;
    eq_expr.lhs() = value;
    eq_expr.rhs() = *it;
    dest.move_to_operands(eq_expr);
  }

  assert(dest.operands().size() != 0);

  if(dest.operands().size() == 1)
  {
    exprt tmp;
    tmp.swap(dest.op0());
    dest.swap(tmp);
  }
}

void goto_convertt::convert_switch(const codet &code, goto_programt &dest)
{
  // switch(v) {
  //   case x: Px;
  //   case y: Py;
  //   ...
  //   default: Pd;
  // }
  // --------------------
  // x: if(v==x) goto X;
  // y: if(v==y) goto Y;
  //    goto d;
  // X: Px;
  // Y: Py;
  // d: Pd;
  // z: ;

  if(code.operands().size() < 2)
  {
    err_location(code);
    throw "switch takes at least two operands";
  }

  exprt argument = code.op0();

  goto_programt sideeffects;
  remove_sideeffects(argument, sideeffects);

  // save break/continue/default/cases targets
  break_continue_switch_targetst old_targets(targets);

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z = tmp_z.add_instruction();
  z->make_skip();
  z->location = code.location();

  // set the new targets -- continue stays as is
  targets.set_break(z);
  targets.set_default(z);
  targets.cases.clear();

  goto_programt tmp;

  forall_operands(it, code)
    if(it != code.operands().begin())
    {
      goto_programt t;
      convert(to_code(*it), t);
      tmp.destructive_append(t);
    }

  goto_programt tmp_cases;

  for(auto &it : targets.cases)
  {
    const caset &case_ops = it.second;

    assert(!case_ops.empty());

    exprt guard_expr;
    case_guard(argument, case_ops, guard_expr);

    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(guard_expr);
      if(globals > 0)
        break_globals2assignments(guard_expr, tmp_cases, code.location());
    }

    goto_programt::targett x = tmp_cases.add_instruction();
    x->make_goto(it.first);
    migrate_expr(guard_expr, x->guard);
    x->location = case_ops.front().find_location();
  }

  {
    goto_programt::targett d_jump = tmp_cases.add_instruction();
    d_jump->make_goto(targets.default_target);
    d_jump->location = targets.default_target->location;
  }

  dest.destructive_append(sideeffects);
  dest.destructive_append(tmp_cases);
  dest.destructive_append(tmp);
  dest.destructive_append(tmp_z);

  // restore old targets
  targets.restore(old_targets);
}

void goto_convertt::convert_break(const code_breakt &code, goto_programt &dest)
{
  if(!targets.break_set)
  {
    err_location(code);
    throw "break without target";
  }

  goto_programt::targett t = dest.add_instruction();
  t->make_goto(targets.break_target);
  t->location = code.location();
}

void goto_convertt::convert_return(
  const code_returnt &code,
  goto_programt &dest)
{
  if(!targets.return_set)
  {
    err_location(code);
    throw "return without target";
  }

  if(code.operands().size() != 0 && code.operands().size() != 1)
  {
    err_location(code);
    throw "return takes none or one operand";
  }

  code_returnt new_code(code);

  if(new_code.has_return_value())
  {
    goto_programt sideeffects;
    remove_sideeffects(new_code.return_value(), sideeffects);
    dest.destructive_append(sideeffects);

    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(new_code.return_value());
      if(globals > 0)
        break_globals2assignments(
          new_code.return_value(), dest, code.location());
    }
  }

  if(targets.return_value)
  {
    if(!new_code.has_return_value())
    {
      err_location(new_code);
      throw "function must return value";
    }
  }
  else
  {
    if(
      new_code.has_return_value() &&
      new_code.return_value().type().id() != "empty")
    {
      err_location(new_code);
      throw "function must not return value";
    }
  }

  goto_programt::targett t = dest.add_instruction();
  t->make_return();
  migrate_expr(new_code, t->code);
  t->location = new_code.location();
}

void goto_convertt::convert_continue(
  const code_continuet &code,
  goto_programt &dest)
{
  if(!targets.continue_set)
  {
    err_location(code);
    throw "continue without target";
  }

  goto_programt::targett t = dest.add_instruction();
  t->make_goto(targets.continue_target);
  t->location = code.location();
}

void goto_convertt::convert_goto(const codet &code, goto_programt &dest)
{
  goto_programt::targett t = dest.add_instruction();
  t->make_goto();
  t->location = code.location();
  migrate_expr(code, t->code);

  // remember it to do target later
  targets.gotos.insert(t);
}

void goto_convertt::convert_non_deterministic_goto(
  const codet &code,
  goto_programt &dest)
{
  convert_goto(code, dest);
}

void goto_convertt::convert_atomic_begin(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 0)
  {
    err_location(code);
    throw "atomic_begin expects no operands";
  }

  copy(code, ATOMIC_BEGIN, dest);
}

void goto_convertt::convert_atomic_end(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 0)
  {
    err_location(code);
    throw "atomic_end expects no operands";
  }

  copy(code, ATOMIC_END, dest);
}

void goto_convertt::generate_ifthenelse(
  const exprt &guard,
  goto_programt &true_case,
  goto_programt &false_case,
  const locationt &location,
  goto_programt &dest)
{
  if(true_case.instructions.empty() && false_case.instructions.empty())
    return;

  // do guarded gotos directly
  if(
    false_case.instructions.empty() && true_case.instructions.size() == 1 &&
    true_case.instructions.back().is_goto() &&
    is_true(true_case.instructions.back().guard))
  {
    migrate_expr(guard, true_case.instructions.back().guard);
    dest.destructive_append(true_case);
    return;
  }

  if(true_case.instructions.empty())
    return generate_ifthenelse(
      gen_not(guard), false_case, true_case, location, dest);

  bool has_else = !false_case.instructions.empty();

  //    if(c) P;
  //--------------------
  // v: if(!c) goto z;
  // w: P;
  // z: ;

  //    if(c) P; else Q;
  //--------------------
  // v: if(!c) goto y;
  // w: P;
  // x: goto z;
  // y: Q;
  // z: ;

  // do the x label
  goto_programt tmp_x;
  goto_programt::targett x = tmp_x.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z = tmp_z.add_instruction();
  z->make_skip();

  // y: Q;
  goto_programt tmp_y;
  goto_programt::targett y;
  if(has_else)
  {
    tmp_y.swap(false_case);
    y = tmp_y.instructions.begin();
  }

  // v: if(!c) goto z/y;
  goto_programt tmp_v;
  generate_conditional_branch(
    gen_not(guard), has_else ? y : z, location, tmp_v);

  // w: P;
  goto_programt tmp_w;
  tmp_w.swap(true_case);

  // x: goto z;
  x->make_goto(z);

  dest.destructive_append(tmp_v);
  dest.destructive_append(tmp_w);

  if(has_else)
  {
    dest.destructive_append(tmp_x);
    dest.destructive_append(tmp_y);
  }

  dest.destructive_append(tmp_z);
}

void goto_convertt::convert_ifthenelse(const codet &code, goto_programt &dest)
{
  if(code.operands().size() != 2 && code.operands().size() != 3)
  {
    err_location(code);
    throw "ifthenelse takes two or three operands";
  }

  bool has_else = code.operands().size() == 3 && !code.op2().is_nil();

  const locationt &location = code.location();

  // convert 'then'-branch
  goto_programt tmp_op1;
  convert(to_code(code.op1()), tmp_op1);

  goto_programt tmp_op2;

  if(has_else)
    convert(to_code(code.op2()), tmp_op2);

  exprt tmp_guard = code.op0();
  remove_sideeffects(tmp_guard, dest);

  generate_ifthenelse(tmp_guard, tmp_op1, tmp_op2, location, dest);
}

void goto_convertt::collect_operands(
  const exprt &expr,
  const irep_idt &id,
  std::list<exprt> &dest)
{
  if(expr.id() != id)
  {
    dest.push_back(expr);
  }
  else
  {
    // left-to-right is important
    forall_operands(it, expr)
      collect_operands(*it, id, dest);
  }
}

void goto_convertt::generate_conditional_branch(
  const exprt &guard,
  goto_programt::targett target_true,
  const locationt &location,
  goto_programt &dest)
{
  if(!has_sideeffect(guard))
  {
    exprt g = guard;
    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(g);
      if(globals > 0)
        break_globals2assignments(g, dest, location);
    }
    // this is trivial
    goto_programt::targett t = dest.add_instruction();
    t->make_goto(target_true);
    migrate_expr(g, t->guard);
    t->location = location;
    return;
  }

  // if(guard) goto target;
  //   becomes
  // if(guard) goto target; else goto next;
  // next: skip;

  goto_programt tmp;
  goto_programt::targett target_false = tmp.add_instruction();
  target_false->make_skip();

  generate_conditional_branch(guard, target_true, target_false, location, dest);

  dest.destructive_append(tmp);
}

void goto_convertt::generate_conditional_branch(
  const exprt &guard,
  goto_programt::targett target_true,
  goto_programt::targett target_false,
  const locationt &location,
  goto_programt &dest)
{
  if(guard.id() == "not")
  {
    assert(guard.operands().size() == 1);
    // swap targets
    generate_conditional_branch(
      guard.op0(), target_false, target_true, location, dest);
    return;
  }

  if(!has_sideeffect(guard))
  {
    exprt g = guard;
    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(g);
      if(globals > 0)
        break_globals2assignments(g, dest, location);
    }

    // this is trivial
    goto_programt::targett t_true = dest.add_instruction();
    t_true->make_goto(target_true);
    migrate_expr(guard, t_true->guard);
    t_true->location = location;

    goto_programt::targett t_false = dest.add_instruction();
    t_false->make_goto(target_false);
    t_false->guard = gen_true_expr();
    t_false->location = location;
    return;
  }

  if(guard.is_and())
  {
    // turn
    //   if(a && b) goto target_true; else goto target_false;
    // into
    //    if(!a) goto target_false;
    //    if(!b) goto target_false;
    //    goto target_true;

    std::list<exprt> op;
    collect_operands(guard, guard.id(), op);

    forall_expr_list(it, op)
      generate_conditional_branch(gen_not(*it), target_false, location, dest);

    goto_programt::targett t_true = dest.add_instruction();
    t_true->make_goto(target_true);
    t_true->guard = gen_true_expr();
    t_true->location = location;

    return;
  }
  if(guard.id() == "or")
  {
    // turn
    //   if(a || b) goto target_true; else goto target_false;
    // into
    //   if(a) goto target_true;
    //   if(b) goto target_true;
    //   goto target_false;

    std::list<exprt> op;
    collect_operands(guard, guard.id(), op);

    forall_expr_list(it, op)
      generate_conditional_branch(*it, target_true, location, dest);

    goto_programt::targett t_false = dest.add_instruction();
    t_false->make_goto(target_false);
    t_false->guard = gen_true_expr();
    t_false->location = guard.location();

    return;
  }

  exprt cond = guard;
  remove_sideeffects(cond, dest);

  if(options.get_bool_option("atomicity-check"))
  {
    unsigned int globals = get_expr_number_globals(cond);
    if(globals > 0)
      break_globals2assignments(cond, dest, location);
  }

  goto_programt::targett t_true = dest.add_instruction();
  t_true->make_goto(target_true);
  migrate_expr(cond, t_true->guard);
  t_true->location = guard.location();

  goto_programt::targett t_false = dest.add_instruction();
  t_false->make_goto(target_false);
  t_false->guard = gen_true_expr();
  t_false->location = guard.location();
}

symbolt &goto_convertt::new_tmp_symbol(const typet &type)
{
  symbolt new_symbol;
  symbolt *symbol_ptr;

  do
  {
    new_symbol.base_name = "tmp$" + i2string(++temporary_counter);
    new_symbol.name = tmp_symbol_prefix + id2string(new_symbol.base_name);
    new_symbol.lvalue = true;
    new_symbol.type = type;
  } while(context.move(new_symbol, symbol_ptr));

  scoped_variables.push_front(symbol_ptr->name);

  return *symbol_ptr;
}
