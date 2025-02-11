#include <cassert>
#include <goto-programs/goto_convert_class.h>
#include <goto-programs/goto_functions.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/location.h>
#include <util/prefix.h>

void goto_convertt::convert_function_call(
  const code_function_callt &function_call,
  goto_programt &dest)
{
  do_function_call(
    function_call.lhs(),
    function_call.function(),
    function_call.arguments(),
    dest);
}

void goto_convertt::do_function_call(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  // make it all side effect free
  exprt new_lhs = lhs, new_function = function;

  exprt::operandst new_arguments = arguments;

  if (!new_lhs.is_nil())
  {
    remove_sideeffects(new_lhs, dest);
  }

  remove_sideeffects(new_function, dest);

  Forall_expr (it, new_arguments)
  {
    remove_sideeffects(*it, dest);
  }

  // split on the function
  if (
    new_function.id() == "dereference" ||
    new_function.id() == "implicit_dereference")
  {
    do_function_call_dereference(new_lhs, new_function, new_arguments, dest);
  }
  else if (new_function.id() == "if")
  {
    do_function_call_if(new_lhs, new_function, new_arguments, dest);
  }
  else if (new_function.id() == "symbol")
  {
    do_function_call_symbol(new_lhs, new_function, new_arguments, dest);
  }
  else if (new_function.id() == "NULL-object")
  {
  }
  else
  {
    log_error("unexpected function argument: {}", new_function.id_string());
    abort();
  }
}

void goto_convertt::do_function_call_if(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if (function.operands().size() != 3)
  {
    log_error("if expects three operands");
    abort();
  }

  // case split

  //    c?f():g()
  //--------------------
  // v: if(!c) goto y;
  // w: f();
  // x: goto z;
  // y: g();
  // z: ;

  // do the v label
  goto_programt tmp_v;
  goto_programt::targett v = tmp_v.add_instruction();

  // do the x label
  goto_programt tmp_x;
  goto_programt::targett x = tmp_x.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z = tmp_z.add_instruction();
  z->make_skip();

  // y: g();
  goto_programt tmp_y;
  goto_programt::targett y;

  do_function_call(lhs, function.op2(), arguments, tmp_y);

  if (tmp_y.instructions.empty())
    y = tmp_y.add_instruction(SKIP);
  else
    y = tmp_y.instructions.begin();

  // v: if(!c) goto y;
  v->make_goto(y);
  migrate_expr(function.op0(), v->guard);
  v->guard = not2tc(v->guard);
  v->location = function.op0().location();

  unsigned int globals = get_expr_number_globals(v->guard);
  if (globals > 1)
  {
    exprt tmp = migrate_expr_back(v->guard);
    break_globals2assignments(tmp, tmp_v, lhs.location());
  }

  // w: f();
  goto_programt tmp_w;

  do_function_call(lhs, function.op1(), arguments, tmp_w);

  if (tmp_w.instructions.empty())
    tmp_w.add_instruction(SKIP);

  // x: goto z;
  x->make_goto(z);

  dest.destructive_append(tmp_v);
  dest.destructive_append(tmp_w);
  dest.destructive_append(tmp_x);
  dest.destructive_append(tmp_y);
  dest.destructive_append(tmp_z);
}

void goto_convertt::do_function_call_dereference(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  goto_programt::targett t = dest.add_instruction(FUNCTION_CALL);

  code_function_callt function_call;
  function_call.location() = function.location();
  function_call.lhs() = lhs;
  function_call.function() = function;
  function_call.arguments() = arguments;

  t->location = function.location();
  migrate_expr(function_call, t->code);
}

void goto_functionst::dump() const
{
  std::ostringstream oss;
  output(*migrate_namespace_lookup, oss);
  log_status("{}", oss.str());
}

void goto_functionst::output(const namespacet &ns, std::ostream &out) const
{
  for (const auto &it : function_map)
  {
    if (it.second.body_available)
    {
      out << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
          << "\n";
      out << "\n";

      const symbolt *symbol = ns.lookup(it.first);
      assert(symbol);
      out << symbol->name << " (" << symbol->id << "):"
          << "\n";
      it.second.body.output(ns, symbol->id, out);
    }
  }
}

void goto_functionst::compute_location_numbers()
{
  unsigned nr = 0;
  for (auto &it : function_map)
    it.second.body.compute_location_numbers(nr);
}

void goto_functionst::compute_target_numbers()
{
  for (auto &it : function_map)
    it.second.body.compute_target_numbers();
}

void goto_functionst::compute_loop_numbers()
{
  unsigned nr = 1;
  for (auto &it : function_map)
    it.second.body.compute_loop_numbers(nr);
}

void get_local_identifiers(
  const goto_functiont &goto_function,
  std::set<irep_idt> &dest)
{
  goto_function.body.get_decl_identifiers(dest);

  const code_typet::argumentst &arguments = goto_function.type.arguments();

  // add parameters
  for (const auto &param : arguments)
  {
    const irep_idt &identifier = param.get_identifier();
    if (identifier != "")
      dest.insert(identifier);
  }
}
