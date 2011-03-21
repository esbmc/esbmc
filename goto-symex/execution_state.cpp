/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include "execution_state.h"
#include <string>
#include <sstream>
#include <i2string.h>
#include <string2array.h>
#include <std_expr.h>
#include <expr_util.h>
#include "../ansi-c/c_types.h"
#include <base_type.h>
#include <simplify_expr.h>
#include <bits/stl_vector.h>
#include "config.h"

#include "basic_symex.h"

//#define DEBUG

unsigned int execution_statet::node_count=0;

/*******************************************************************
 Function: execution_statet::get_active_state

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

goto_symex_statet & execution_statet::get_active_state() {
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

    return _threads_state.at(_active_thread);
}

/*******************************************************************
 Function: execution_statet::all_threads_ended

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool execution_statet::all_threads_ended()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  for(unsigned int i=0; i < _threads_state.size();i++)
    if(!_threads_state.at(i).thread_ended)
      return false;
  return true;
}

/*******************************************************************
 Function: execution_statet::get_active_atomic_number

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

unsigned int execution_statet::get_active_atomic_number()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return _atomic_numbers.at(_active_thread);
}

/*******************************************************************
 Function: execution_statet::increment_active_atomic_number

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::increment_active_atomic_number()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  _atomic_numbers.at(_active_thread)++;
}

/*******************************************************************
 Function: execution_statet::decrement_active_atomic_number

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::decrement_active_atomic_number()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  _atomic_numbers.at(_active_thread)--;
//  assert(_atomic_numbers.at(_active_thread) >= 0);
}

/*******************************************************************
 Function: execution_statet::get_guard_identifier

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

irep_idt execution_statet::get_guard_identifier()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return id2string(guard_execution) + '@' + i2string(_CS_number) + '_' + i2string(_last_active_thread) + '_' + i2string(node_id) + '&' + i2string(node_id) + "#1";
}

/*******************************************************************
 Function: execution_statet::get_guard_identifier_base

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

irep_idt execution_statet::get_guard_identifier_base()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return id2string(guard_execution) + '@' + i2string(_CS_number) + '_' + i2string(_last_active_thread) + '_' + i2string(node_id);
}


/*******************************************************************
 Function: execution_statet::set_parent_guard

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::set_parent_guard(const irep_idt & parent_guard)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  _parent_guard_identifier = parent_guard;
}

/*******************************************************************
 Function: execution_statet::set_active_stat

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::set_active_state(unsigned int i)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  _last_active_thread = _active_thread;
  _active_thread = i;
}

/*******************************************************************
 Function: execution_statet::decreament_trds_in_run

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::decreament_trds_in_run(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  typet int_t = int_type();
  exprt one_expr = gen_one(int_t);
  exprt lhs_expr = symbol_exprt("c::trds_in_run", int_t);
  exprt op1 = lhs_expr;
  exprt rhs_expr = gen_binary("-", int_t, op1, one_expr);

  get_active_state().rename(rhs_expr, ns,node_id);
  base_type(rhs_expr, ns);
  simplify(rhs_expr);

  exprt new_lhs = lhs_expr;

  get_active_state().assignment(new_lhs, rhs_expr, ns, true, *this, node_id);

  target.assignment(
		  get_active_state().guard,
          new_lhs, lhs_expr,
          rhs_expr,
          get_active_state().source,
          symex_targett::STATE);
}

/*******************************************************************
 Function: execution_statet::end_thread

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::end_thread(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

    get_active_state().thread_ended = true;

    decreament_trds_in_run(ns,target);
#if 1
    typet uint_t = uint_type();


    unsigned int mask_num = ~(_active_thread << 1);

    constant_exprt mask_num_expr = constant_exprt(uint_t);
    mask_num_expr.set_value(integer2binary(mask_num, config.ansi_c.int_width));

    exprt lhs_expr = symbol_exprt("c::trds_status", uint_t);
    exprt op1 = lhs_expr;
    exprt rhs_expr = gen_binary("bitand", uint_t, op1, mask_num_expr);

    get_active_state().rename(rhs_expr, ns,node_id);
    base_type(rhs_expr, ns);
    simplify(rhs_expr);

    exprt new_lhs = lhs_expr;

    get_active_state().assignment(new_lhs, rhs_expr, ns, true, this, node_id);

    target.assignment(
            get_active_state().guard,
            new_lhs, lhs_expr,
            rhs_expr,
            get_active_state().source,
            symex_targett::STATE);
#endif
}

/*******************************************************************
 Function: execution_statet::increament_trds_in_run

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::increament_trds_in_run(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  static bool thrds_in_run_flag=1;
  typet int_t = int_type();

  if (thrds_in_run_flag)
  {
    exprt lhs_expr = symbol_exprt("c::trds_in_run", int_t);
    constant_exprt rhs_expr(int_t);
    rhs_expr.set_value(integer2binary(1,config.ansi_c.int_width));

    get_active_state().assignment(lhs_expr, rhs_expr, ns, true, *this, node_id);

    thrds_in_run_flag=0;
  }

  exprt one_expr = gen_one(int_t);
  exprt lhs_expr = symbol_exprt("c::trds_in_run", int_t);
  exprt op1 = lhs_expr;
  exprt rhs_expr = gen_binary("+", int_t, op1, one_expr);

  get_active_state().rename(rhs_expr, ns, node_id);
  base_type(rhs_expr, ns);
  simplify(rhs_expr);

  exprt new_lhs = lhs_expr;

  get_active_state().assignment(new_lhs, rhs_expr, ns, true, *this, node_id);

  target.assignment(
          get_active_state().guard,
          new_lhs, lhs_expr,
          rhs_expr,
          get_active_state().source,
          symex_targett::STATE);
}

/*******************************************************************
 Function: execution_statet::deadlock_detection

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/
#if 0
void execution_statet::deadlock_detection(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  static bool deadlock_detection_flag=0;

  if (deadlock_detection_flag)
	return ;

  typet int_t = int_type();

  //exprt one_expr = gen_one(int_t);
  exprt lhs_expr = symbol_exprt("c::deadlock_detection", int_t);
  exprt op1 = lhs_expr;
  constant_exprt rhs_expr(int_t);

  if (_no_deadlock_detection)
    rhs_expr.set_value(integer2binary(1,config.ansi_c.int_width));
  else
    rhs_expr.set_value(integer2binary(0,config.ansi_c.int_width));

  get_active_state().rename(rhs_expr, ns, node_id);
  base_type(rhs_expr, ns);
  simplify(rhs_expr);

  exprt new_lhs = lhs_expr;

  get_active_state().assignment(new_lhs, rhs_expr, ns, true, this, node_id);

  target.assignment(
          get_active_state().guard,
          new_lhs, lhs_expr,
          rhs_expr,
          get_active_state().source,
          symex_targett::STATE);

  deadlock_detection_flag=1;

}
#endif
/*******************************************************************
 Function:  execution_statet::update_trds_count

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::update_trds_count(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  typet int_t = int_type();
  exprt lhs_expr = symbol_exprt("c::trds_count", int_t);
  exprt op1 = lhs_expr;

  constant_exprt rhs_expr = constant_exprt(int_t);
  rhs_expr.set_value(integer2binary(_threads_state.size()-1, config.ansi_c.int_width));
  get_active_state().rename(rhs_expr, ns,node_id);
  base_type(rhs_expr, ns);
  simplify(rhs_expr);

  exprt new_lhs = lhs_expr;

  get_active_state().assignment(new_lhs, rhs_expr, ns, true, *this, node_id);

  target.assignment(
          get_active_state().guard,
          new_lhs, lhs_expr,
          rhs_expr,
          get_active_state().source,
          symex_targett::STATE);
}

/*******************************************************************
 Function: execution_statet::update_trds_status

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/
#if 1
void execution_statet::update_trds_status(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

	typet uint_t = uint_type();

    unsigned int mask_num = (_threads_state.size()-1) << 1;

    constant_exprt mask_num_expr = constant_exprt(uint_t);
    mask_num_expr.set_value(integer2binary(mask_num, config.ansi_c.int_width));

    exprt lhs_expr = symbol_exprt("c::trds_status", uint_t);
    exprt op1 = lhs_expr;
    exprt rhs_expr = gen_binary("bitor", uint_t, op1, mask_num_expr);

    get_active_state().rename(rhs_expr, ns,node_id);
    base_type(rhs_expr, ns);
    simplify(rhs_expr);

    exprt new_lhs = lhs_expr;

    get_active_state().assignment(new_lhs, rhs_expr, ns, true, this, node_id);

    target.assignment(
            get_active_state().guard,
            new_lhs, lhs_expr,
            rhs_expr,
            get_active_state().source,
            symex_targett::STATE);

}
#endif

/*******************************************************************
 Function: execution_statet::set_trd_number

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::set_active_trd_zero(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

	typet uint_t = uint_type();

	//std::cout << "set_active_trd _active_thread: " << _active_thread << std::endl;
	//std::cout << "_active_thread: " << _threads_state.size()-1 << std::endl;

	exprt lhs_expr = symbol_exprt("c::trd_nr", uint_t);
    constant_exprt rhs_expr = constant_exprt(uint_t);
    rhs_expr.set_value(integer2binary(_threads_state.size()-1, config.ansi_c.int_width));

    exprt new_lhs = lhs_expr;

    get_active_state().assignment(new_lhs, rhs_expr, ns, true,node_id);

    target.assignment(
            get_active_state().guard,
            new_lhs, lhs_expr,
            rhs_expr,
            get_active_state().source,
            symex_targett::STATE);

}

/*******************************************************************
 Function: execution_statet::set_trd_number

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::set_active_trd_one(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

	typet uint_t = uint_type();

	//std::cout << "set_active_trd _active_thread: " << _active_thread << std::endl;
	std::cout << "_active_thread: " << _active_thread << std::endl;

	exprt lhs_expr = symbol_exprt("c::trd_nr", uint_t);
    constant_exprt rhs_expr = constant_exprt(uint_t);
    rhs_expr.set_value(integer2binary(_active_thread, config.ansi_c.int_width));

    exprt new_lhs = lhs_expr;

    get_active_state().assignment(new_lhs, rhs_expr, ns, true,node_id);

    target.assignment(
            get_active_state().guard,
            new_lhs, lhs_expr,
            rhs_expr,
            get_active_state().source,
            symex_targett::STATE);

}

/*******************************************************************
 Function: execution_statet::set_trd_status_to_one

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::set_trd_status_to_one(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

#if 1
  typet uint_t = uint_type();

  irep_idt tmp = "c::exit" + i2string(_active_thread);
  exprt lhs_expr = symbol_exprt(tmp, uint_t);
  exprt rhs_expr = gen_one(uint_t);

  exprt new_lhs = lhs_expr;

  get_active_state().assignment(new_lhs, rhs_expr, ns, true,node_id);

  target.assignment(
          get_active_state().guard,
          new_lhs, lhs_expr,
          rhs_expr,
          get_active_state().source,
          symex_targett::STATE);
#endif
}

/*******************************************************************
 Function: execution_statet::set_trd_status_to_zero

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::set_trd_status_to_zero(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

#if 1
  typet uint_t = uint_type();

  irep_idt tmp = "c::exit" + i2string(_threads_state.size()-1);
  exprt lhs_expr = symbol_exprt(tmp, uint_t);
  exprt rhs_expr = gen_zero(uint_t);

  exprt new_lhs = lhs_expr;

  get_active_state().assignment(new_lhs, rhs_expr, ns, true,node_id);

  target.assignment(
          get_active_state().guard,
          new_lhs, lhs_expr,
          rhs_expr,
          get_active_state().source,
          symex_targett::STATE);
#endif
}


/*******************************************************************
 Function: execution_statet::execute_guard

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::execute_guard(const namespacet &ns, symex_targett &target)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

    parent_node_id = node_id;
    node_id = node_count++;
    exprt guard_expr = symbol_exprt(get_guard_identifier_base(), bool_typet());
    exprt parent_guard;
    exprt new_lhs = guard_expr;

    typet my_type = uint_type();
    //exprt trd_expr = symbol_exprt(id2string(guard_thread) + i2string(_CS_number), my_type);
    exprt trd_expr = symbol_exprt(get_guard_identifier_base(), my_type);
    constant_exprt num_expr = constant_exprt(my_type);
    num_expr.set_value(integer2binary(_active_thread, config.ansi_c.int_width));
    exprt cur_rhs = equality_exprt(trd_expr, num_expr);

    exprt new_rhs;
    parent_guard = true_exprt();
    new_rhs = parent_guard;

    if (!_parent_guard_identifier.empty())
    {
      parent_guard = symbol_exprt(_parent_guard_identifier, bool_typet());
      new_rhs = cur_rhs; //gen_and(parent_guard, cur_rhs);
    }

    get_active_state().assignment(new_lhs, new_rhs, ns, false, *this, node_id);

    assert(new_lhs.get("identifier") == get_guard_identifier());

    guardt old_guard;
    old_guard.add(parent_guard);
    exprt new_guard_expr = symbol_exprt(get_guard_identifier(), bool_typet());

    guardt guard;
    target.assignment(
            guard,
            new_lhs, guard_expr,
            new_rhs,
            get_active_state().source,
            symex_targett::HIDDEN);

    // copy the new guard exprt to every threads
    for (unsigned int i = 0; i < _threads_state.size(); i++)
    {
      // remove the old guard first
      _threads_state.at(i).guard -= old_guard;
      _threads_state.at(i).guard.add(new_guard_expr);
    }
}

/*******************************************************************
 Function: execution_statet::add_thread

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::add_thread(goto_programt::const_targett thread_start, goto_programt::const_targett thread_end)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  goto_symex_statet state;
  state.initialize(_state_level2, thread_start, thread_end, _threads_state.size());

  _threads_state.push_back(state);
  _atomic_numbers.push_back(0);

  _DFS_traversed.push_back(false);
  _exprs.push_back(exprt());

  _exprs_read_write.push_back(read_write_set());
}

/*******************************************************************
 Function: execution_statet::add_thread

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void execution_statet::add_thread(goto_symex_statet & state)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  goto_symex_statet new_state(state);

  new_state.source.thread_nr = _threads_state.size();
  _threads_state.push_back(new_state);
  _atomic_numbers.push_back(0);

  _DFS_traversed.push_back(false);
  _exprs.push_back(exprt());

  _exprs_read_write.push_back(read_write_set());
}


/*******************************************************************
 Function: execution_statet::is_in_lookup

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool execution_statet::is_in_lookup(const namespacet &ns, const irep_idt &identifier) const
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  const symbolt *symbol;

  if (ns.lookup(identifier, symbol))
    return true;

  return false;
}

/*******************************************************************
 Function: execution_statet::lookup

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

const symbolt &execution_statet::lookup(const namespacet &ns, const irep_idt &identifier) const
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  const symbolt *symbol;

  if (ns.lookup(identifier, symbol))
    throw "failed to find symbol " + id2string(identifier);

  return *symbol;
}

/*******************************************************************
 Function: execution_statet::get_expr_write_globals

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

unsigned int execution_statet::get_expr_write_globals(const namespacet &ns, const exprt & expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << "\n";
#endif

  std::string identifier = expr.get_string("identifier");

  if (expr.id() == "address_of" ||
      expr.id() == "valid_object" ||
      expr.id() == "dynamic_size" ||
      expr.id() == "dynamic_type" ||
      expr.id() == "is_zero_string" ||
      expr.id() == "zero_string" ||
      expr.id() == "zero_string_length")
      return 0;
  else if (expr.id() == "symbol")
  {
    const irep_idt &id = expr.get("identifier");
    const irep_idt &identifier = get_active_state().get_original_name(id);
    const symbolt &symbol = lookup(ns, identifier);

    if (identifier == "c::__ESBMC_alloc"
        || identifier == "c::__ESBMC_alloc_size")
      return 0;
    else if ((symbol.static_lifetime || symbol.type.get("#dynamic") != ""))
    {
      //std::cout << "get_expr_write_globals: " << expr.pretty() << std::endl;
      _exprs_read_write.at(_active_thread).write_set.insert(identifier);
      return 1;
    }
    else
      return 0;
  }
#if 0
  else if (expr.id() == "index")
  {
	if (expr.op0().id() == "symbol" && expr.op1().id()=="constant")
	{
      const irep_idt &id = expr.op0().get("identifier");
	  const irep_idt &identifier = get_active_state().get_original_name(id);
	  const symbolt &symbol = lookup(ns, identifier);

	  if ((symbol.static_lifetime || symbol.type.get("#dynamic") != ""))
	  {
	    std::string value, array_name;
	    value = integer2string(binary2integer(expr.op1().get_string("value"), true),10);
	    array_name = expr.op0().get_string("identifier") + "::" + value;
	    const irep_idt &array_identifier = array_name;
	    //std::cout << "write_globals identifier: " << array_identifier.c_str() << std::endl;
	    _exprs_read_write.at(_active_thread).write_set.insert(array_identifier);
	    return 1;
	  }
	  else
		return 0;
	}
  }

  else if (expr.id() == "member")
  {
      const irep_idt &id = expr.op0().get("identifier");
	  const irep_idt &identifier = get_active_state().get_original_name(id);
	  const symbolt &symbol = lookup(ns, identifier);

	  if ((symbol.static_lifetime || symbol.type.get("#dynamic") != ""))
	  {
	    std::string value, array_name;
	    value = expr.get_string("component_name");
	    array_name = expr.op0().get_string("identifier") + "::" + value;
	    const irep_idt &array_identifier = array_name;
	    std::cout << "write_globals identifier: " << array_identifier.c_str() << std::endl;
	    _exprs_read_write.at(_active_thread).write_set.insert(array_identifier);
	    return 1;
	  }
	  else
		return 0;
  }
#endif

    unsigned int globals = 0;

    forall_operands(it, expr) {
        globals += get_expr_write_globals(ns, *it);
    }

  return globals;
}

/*******************************************************************
 Function: execution_statet::get_expr_read_globals

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

unsigned int execution_statet::get_expr_read_globals(const namespacet &ns, const exprt & expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << "\n";
#endif

  std::string identifier = expr.get_string("identifier");

  if (expr.id() == "address_of" ||
            expr.type().id() == "pointer" ||
            expr.id() == "valid_object" ||
            expr.id() == "dynamic_size" ||
            expr.id() == "dynamic_type" ||
            expr.id() == "is_zero_string" ||
            expr.id() == "zero_string" ||
            expr.id() == "zero_string_length")
    return 0;
  else if (expr.id() == "symbol")
  {
    const irep_idt &id = expr.get("identifier");
    const irep_idt &identifier = get_active_state().get_original_name(id);

    if (identifier == "goto_symex::\\guard!" + i2string(get_active_state().top().level1._thread_id))
      return 0;

    if (is_in_lookup(ns, identifier))
      return 0;

    const symbolt &symbol = lookup(ns, identifier);

    if (identifier == "c::__ESBMC_alloc" || identifier == "c::__ESBMC_alloc_size")
      return 0;
    else if ((symbol.static_lifetime || symbol.type.get("#dynamic") != ""))
    {
      //std::cout << "get_expr_read_globals: " << expr.pretty() << std::endl;
      _exprs_read_write.at(_active_thread).read_set.insert(identifier);
      return 1;
    }
    else
      return 0;
  }
#if 0
  else if (expr.id() == "index")
  {
	if (expr.op0().id() == "symbol" && expr.op1().id()=="constant")
	{
      const irep_idt &id = expr.op0().get("identifier");
	  const irep_idt &identifier = get_active_state().get_original_name(id);
	  const symbolt &symbol = lookup(ns, identifier);

	  if ((symbol.static_lifetime || symbol.type.get("#dynamic") != ""))
	  {
	    std::string value, array_name;
        value = integer2string(binary2integer(expr.op1().get_string("value"), true),10);
        array_name = expr.op0().get_string("identifier") + "::" + value;
        const irep_idt &array_identifier = array_name;
        //std::cout << "read_globals identifier: " << array_identifier.c_str() << std::endl;
        _exprs_read_write.at(_active_thread).read_set.insert(array_identifier);
        return 1;
	  }
	  else
	    return 0;
	}
  }

  else if (expr.id() == "member")
  {
      const irep_idt &id = expr.op0().get("identifier");
	  const irep_idt &identifier = get_active_state().get_original_name(id);
	  const symbolt &symbol = lookup(ns, identifier);

	  if ((symbol.static_lifetime || symbol.type.get("#dynamic") != ""))
	  {
	    std::string value, array_name;
        value = expr.get_string("component_name");
        array_name = expr.op0().get_string("identifier") + "::" + value;
        const irep_idt &array_identifier = array_name;
        std::cout << "read_globals identifier: " << array_identifier.c_str() << std::endl;
        _exprs_read_write.at(_active_thread).read_set.insert(array_identifier);
        return 1;
	  }
	  else
	    return 0;
  }
#endif
  unsigned int globals = 0;

  forall_operands(it, expr) {
    globals += get_expr_read_globals(ns, *it);
  }

  return globals;
}

crypto_hash
execution_statet::generate_hash(void) const
{

  crypto_hash state = _state_level2->generate_l2_state_hash();
  std::string str = state.to_string();

  for (std::vector<goto_symex_statet>::const_iterator it=_threads_state.begin();
        it != _threads_state.end(); it++) {
    goto_programt::const_targett pc = it->source.pc;
    int id = pc->location_number;
    str += "!" + id;
  }

  crypto_hash h = crypto_hash(str);

  return h;
}

void
execution_statet::extract_elems_from_array_expr(exprt array,
                        std::map<std::string,std::string> &m)
{

  if (array.op0().id() == "with") {
    extract_elems_from_array_expr(array.op0(), m);
  } else {
    m["__ESBMC_hash_array_base"] = serialise_expr(array.op0());
  }

  std::string idx = serialise_expr(array.op1());
  std::string val = serialise_expr(array.op2());
  m[idx] = val;
  return;
}

void
execution_statet::extract_struct_members_from_expr(exprt st,
                        std::map<std::string,std::string> &m)
{

  if (st.op0().id() == "with") {
    extract_struct_members_from_expr(st.op0(), m);
  } else {
    m["__ESBMC_hash_struct_base"] = serialise_expr(st.op0());
  }

  std::string member = serialise_expr(st.op1());
  std::string val = serialise_expr(st.op2());
  m[member] = val;
}

static std::string state_to_ignore[8] =
{"\\guard_exec", "trds_count", "trds_in_run", "deadlock_wait", "deadlock_mutex",
"count_lock", "count_wait", "unlocked"};

std::string
execution_statet::serialise_expr(const exprt &rhs)
{
  std::string str;
  uint64_t val;
  int i;

  // FIXME: some way to disambiguate what's part of a hash / const /whatever,
  // and what's part of an operator

  // The plan: serialise this expression into the identifiers of its operations,
  // replacing symbol names with the hash of their value.
  if (rhs.id() == "symbol") {

    str = rhs.get("identifier").as_string();
    for (i = 0 ; i < 8; i++)
      if (str.find(state_to_ignore[i]) != std::string::npos)
        return "(ignore)";

    exprt bees = _target.reconstruct_expr_from_SSA(rhs);
    /* Now then, is this a reconstructed expr, or could we get no further? */
    if (bees.id() == "symbol") {
      /* Reached something assumed or nondeterministic: just return the symbol
       * name */
      return bees.get("identifier").as_string();
    }

    /* Otherwise, we can continue munging */
    return serialise_expr(bees);
  } else if (rhs.id() == "array_of") {
    /* An array of the same set of values: generate all of them. */
    str = "array(";
    irept array = rhs.find("type");
    irept size = array.find("size");
    std::string sz = size.get("value").as_string();
    val = strtol(sz.c_str(), NULL, 2);
    for (i = 0; i < val; i++)
      str += "elem(" + serialise_expr(rhs.op0()) + "),";
  } else if (rhs.id() == "with") {
    exprt bees = _target.reconstruct_expr_from_SSA(rhs);
    if (rhs.op0().type().id() == "array") {
      std::map<std::string,std::string> m;
      extract_elems_from_array_expr(bees, m);

      str = "array(";
      std::map<std::string,std::string>::const_iterator it;
      for (it = m.begin(); it != m.end(); it++)
        str += "(idx(" + it->first + "),val(" + it->second + ")),";
    } else if (rhs.op0().type().id() == "struct") {
      std::map<std::string,std::string> m;
      extract_struct_members_from_expr(bees, m);

      str = "struct(";
      std::map<std::string,std::string>::const_iterator it;
      for (it = m.begin(); it != m.end(); it++)
        str += "(member(" + it->first + "),val(" + it->second + ")),";
    } else if (rhs.op0().type().id() ==  "union") {
      /* We don't care about previous assignments to this union, because
       * they're overwritten by this one, and leads to undefined side effects
       * anyway. So, just serialise the identifier, the member assigned to,
       * and the value assigned */
      assert(rhs.op0().id() == "symbol");
      str = "union_set(";
      str += "union_sym(" + rhs.op0().get("identifier").as_string() + "),";
      str += "field(" + serialise_expr(rhs.op1()) + "),";
      str += "val(" + serialise_expr(rhs.op2()) + "))";
    } else {
      throw "Unrecognised type of with expression: " + rhs.op0().type().id().as_string();
    }
  } else if (rhs.id() == "address_of") {
    str = "addressof(" + serialise_expr(rhs.op0()) + ")";
  } else if (rhs.id() == "index") {
    str = "index(";
    str += serialise_expr(rhs.op0());
    str += ",idx(" + serialise_expr(rhs.op1()) + ")";
  } else if (rhs.id() == "typecast") {
    str = "typecast(" + serialise_expr(rhs.op0()) + ")";
  } else if (rhs.id() == "pointer_object") {
    str = "pointer_obj(" + serialise_expr(rhs.op0()) + ")";
  } else if (rhs.id() == "member_name") {
    str = "component(" + rhs.get("component_name").as_string() + ")";
  } else if (rhs.id() == "member") {
    str = "member(entity(" + serialise_expr(rhs.op0()) + "),";
    str += "member_name(" + rhs.get("component_name").as_string() + "))";
  } else if (rhs.id() == "nondet_symbol") {
    /* Just return the identifier: it'll be unique to this particular piece
     * of entropy */
    str = "nondet_symbol(" + rhs.get("identifier").as_string() + ")";
  } else if (rhs.id() == "if") {
    str = "cond(if(" + serialise_expr(rhs.op0()) + "),";
    str += "then(" + serialise_expr(rhs.op1()) + "),";
    str += "else(" + serialise_expr(rhs.op2()) + "))";
  } else if (rhs.id() == "struct") {
    str = rhs.type().get("tag").as_string();
    str = "struct(tag(" + str + "),";
    forall_operands(it, rhs) {
      str = str + "(" + serialise_expr(*it) + "),";
    }
    str += ")";
  } else if (rhs.id() == "union") {
    str = rhs.type().get("tag").as_string();
    str = "union(tag(" + str + "),";
    forall_operands(it, rhs) {
      str = str + "(" + serialise_expr(*it) + "),";
    }
  } else if (rhs.id() == "constant") {
    // It appears constants can be "true", "false", or a bit vector. Parse that,
    // and then print the value as a base 10 integer.

    irep_idt idt_val = rhs.get("value");
    if (idt_val == "true") {
      val = 1;
    } else if (idt_val == "false") {
      val = 0;
    } else {
      val = strtol(idt_val.c_str(), NULL, 2);
    }

    std::stringstream tmp;
    tmp << val;
    str = "const(" + tmp.str() + ")";

  // If we have a normal expression, either arithmatic, binary, comparision,
  // or whatever, just take the operator and append its operands.
  } else if (rhs.id() == "+" || rhs.id() == "-" || rhs.id() == "*" || rhs.id() == "/" || rhs.id() == "mod" || rhs.id() == "=" || rhs.id() == "and" || rhs.id() == "=>" || rhs.id() == "or" || rhs.id() == "not" || rhs.id() == "notequal" || rhs.id() ==  "unary-" || rhs.id() == "abs" || rhs.id() == ">=" || rhs.id() == ">" || rhs.id() == "<=" || rhs.id() == "<" || rhs.id() == "bitor" || rhs.id() == "bitxor" || rhs.id() == "bitand" || rhs.id() == "bitnot" || rhs.id() == "shl" || rhs.id() == "lshr" || rhs.id() == "ashr") {
    str = rhs.id().as_string();
    forall_operands(it, rhs) {
      str = str + "(" + serialise_expr(*it) + ")";
    }
  } else if (rhs.id() == "pointer_offset") {
    str = "pointer_offset(" + serialise_expr(rhs.op0()) + ")";
  } else if (rhs.id() == "string-constant") {
    exprt tmp;
    string2array(rhs, tmp);
    return serialise_expr(tmp);
  } else {
    std::cout << "Unrecognized expression when generating state hash:\n";
    std::cout << rhs.pretty(0) << std::endl;
    abort();
  }

  return str;
}

crypto_hash
execution_statet::update_hash_for_assignment(const exprt &rhs)
{

  return crypto_hash(serialise_expr(rhs));
}
