/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com
		Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>
#include <global.h>
#include <malloc.h>
#include <map>
#include <sstream>

#include <i2string.h>
#include "../util/expr_util.h"

#include "reachability_tree.h"
#include "execution_state.h"
#include "goto_symex_state.h"
#include "goto_symex.h"
#include "crypto_hash.h"

/*******************************************************************\

Function: goto_symex_statet::initialize

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symex_statet::initialize(const goto_programt::const_targett & start, const goto_programt::const_targett & end, const goto_programt *prog, unsigned int thread_id)
{
  new_frame(thread_id);

  source.is_set=true;
  source.thread_nr = thread_id;
  source.pc=start;
  top().end_of_function=end;
  top().calling_location=symex_targett::sourcet(top().end_of_function, prog);
}

/*******************************************************************\

Function: goto_symex_statet::constant_propagation

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symex_statet::constant_propagation(const exprt &expr) const
{
  static unsigned int with_counter=0;
  if(expr.is_constant()) return true;

  if(expr.id()==exprt::addrof)
  {
    if(expr.operands().size()!=1)
      throw "address_of expects one operand";

    return constant_propagation_reference(expr.op0());
  }
  else if(expr.id()==exprt::typecast)
  {
    if(expr.operands().size()!=1)
      throw "typecast expects one operand";

    return constant_propagation(expr.op0());
  }
  else if(expr.id()==exprt::plus)
  {
    forall_operands(it, expr)
      if(!constant_propagation(*it))
        return false;

    return true;
  }
#if 1
  else if(expr.id()==exprt::arrayof)
  {
    if(expr.operands().size()==1)
      if (expr.op0().id()==exprt::constant && expr.op0().type().id()!=typet::t_bool)
        return true;
  }
#endif
#if 1
  else if(expr.id()==exprt::with)
  {
	with_counter++;

	if (with_counter>6)
	{
		with_counter=0;
		return false;
	}

    //forall_operands(it, expr)
    //{
      if(!constant_propagation(expr.op0()))
      {
    	with_counter=0;
        return false;
      }
    //}
    with_counter=0;
    return true;
  }
#endif
  else if(expr.id()=="struct")
  {
    forall_operands(it, expr)
      if(!constant_propagation(expr.op0()))
        return false;

    return true;
  }

  else if(expr.id()=="union")
  {
    if(expr.operands().size()==1)
      return constant_propagation(expr.op0());
  }

  /* No difference
  else if(expr.id()==exprt::equality)
  {
    if(expr.operands().size()!=2)
	  throw "equality expects two operands";

    return (constant_propagation(expr.op0()) ||
           constant_propagation(expr.op1()));

  }
  */

  return false;
}

/*******************************************************************\

Function: goto_symex_statet::constant_propagation_reference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symex_statet::constant_propagation_reference(const exprt &expr) const
{
  if(expr.id()==exprt::symbol)
    return true;
  else if(expr.id()==exprt::index)
  {
    if(expr.operands().size()!=2)
      throw "index expects two operands";

    return constant_propagation_reference(expr.op0()) &&
           constant_propagation(expr.op1());
  }
  else if(expr.id()==exprt::member)
  {
    if(expr.operands().size()!=1)
      throw "member expects one operand";

    return constant_propagation_reference(expr.op0());
  }
#if 1
  else if(expr.id()=="string-constant")
    return true;
#endif

  return false;
}

/*******************************************************************\

Function: goto_symex_statet::assignment

  Inputs:

 Outputs:

 Purpose: write to a variable

\*******************************************************************/

void goto_symex_statet::assignment(
  exprt &lhs,
  const exprt &rhs,
  const namespacet &ns,
  bool record_value,
  execution_statet &ex_state,
  unsigned exec_node_id)
{
  crypto_hash hash;
  assert(lhs.id()=="symbol");
  assert(lhs.id()==exprt::symbol);

  if (ex_state.owning_rt->state_hashing)
    hash = ex_state.update_hash_for_assignment(rhs);

  // the type might need renaming
  rename(lhs.type(), ns, exec_node_id);

  const irep_idt &identifier= lhs.identifier();

  // identifier should be l0 or l1, make sure it's l1

  const std::string l1_identifier=top().level1(identifier,exec_node_id);
  std::string orig_name = get_original_name(l1_identifier).as_string();

  // do the l2 renaming
  level2t::valuet &entry=level2.current_names[l1_identifier];

  entry.count++;

  level2.rename(l1_identifier, entry.count,exec_node_id);

  lhs.identifier(level2.name(l1_identifier, entry.count));

  if (ex_state.owning_rt->state_hashing)
    level2.current_hashes[orig_name] = hash;

  if(record_value)
  {
    // for constant propagation

    if(constant_propagation(rhs))
      entry.constant=rhs;
    else
      entry.constant.make_nil();
  }
  else
    entry.constant.make_nil();

  if(use_value_set)
  {
    // update value sets
    value_sett::expr_sett rhs_value_set;
    exprt l1_rhs(rhs);
    level2.get_original_name(l1_rhs);

    exprt l1_lhs(exprt::symbol, lhs.type());
    l1_lhs.identifier(l1_identifier);

    value_set.assign(l1_lhs, l1_rhs, ns);
  }
}

/*******************************************************************\

Function: goto_symex_statet::rename

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symex_statet::rename(exprt &expr, const namespacet &ns,unsigned node_id)
{
  // rename all the symbols with their last known value

  rename(expr.type(), ns,node_id);

  if(expr.id()==exprt::symbol)
  {
    top().level1.rename(expr,node_id);
    level2.rename(expr,node_id);
  }
  else if(expr.id()==exprt::addrof ||
          expr.id()=="implicit_address_of" ||
          expr.id()=="reference_to")
  {
    assert(expr.operands().size()==1);
    rename_address(expr.op0(), ns,node_id);
  }
  else
  {
    // do this recursively
    Forall_operands(it, expr)
      rename(*it, ns,node_id);
  }
}

/*******************************************************************\

Function: goto_symex_statet::rename_address

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symex_statet::rename_address(
  exprt &expr,
  const namespacet &ns, unsigned node_id)
{
  // rename all the symbols with their last known value

  rename(expr.type(), ns,node_id);

  if(expr.id()==exprt::symbol)
  {
    // only do L1
    top().level1.rename(expr,node_id);
  }
  else if(expr.id()==exprt::index)
  {
    assert(expr.operands().size()==2);
    rename_address(expr.op0(), ns,node_id);
    rename(expr.op1(), ns,node_id);
  }
  else
  {
    // do this recursively
    Forall_operands(it, expr)
      rename_address(*it, ns,node_id);
  }
}

/*******************************************************************\

Function: goto_symex_statet::rename

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symex_statet::rename(
  typet &type,
  const namespacet &ns, unsigned node_id)
{
  // rename all the symbols with their last known value

  if(type.id()==typet::t_array)
  {
    rename(type.subtype(), ns,node_id);
    exprt tmp = static_cast<const exprt &>(type.size_irep());
    rename(tmp, ns,node_id);
    type.size(tmp);
  }
  else if(type.id()==typet::t_struct ||
          type.id()==typet::t_union ||
          type.id()==typet::t_class)
  {
    // TODO
  }
  else if(type.id()==typet::t_pointer)
  {
    // rename(type.subtype(), ns);
    // don't do this, or it might get cyclic
  }
  else if(type.id()==exprt::symbol)
  {
	const symbolt &symbol=ns.lookup(type.identifier());
	type=symbol.type;
    rename(type, ns,node_id);
  }
}
/*******************************************************************\

Function: goto_symex_statet::get_original_name

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symex_statet::get_original_name(exprt &expr) const
{
  Forall_operands(it, expr)
    get_original_name(*it);

  if(expr.id()==exprt::symbol)
  {
    level2.get_original_name(expr);
    top().level1.get_original_name(expr);
  }
}

/*******************************************************************\

Function: goto_symex_statet::get_original_identifier

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const irep_idt &goto_symex_statet::get_original_name(
  const irep_idt &identifier) const
{

  return top().level1.get_original_name(
         level2.get_original_name(identifier));
}

void goto_symex_statet::print_stack_trace(const namespacet &ns, unsigned int indent) const
{
  call_stackt::const_reverse_iterator it;
  symex_targett::sourcet src;
  std::string spaces = std::string("");
  int i;

  for (i = 0; i < indent; i++)
    spaces += " ";

  // Iterate through each call frame printing func name and location.
  src = source;
  for (it = call_stack.rbegin(); it != call_stack.rend(); it++) {
    if (it->function_identifier == "") { // Top level call
      std::cout << spaces << "init" << std::endl;
    } else {
      std::cout << spaces << it->function_identifier.as_string();
      std::cout << " at " << src.pc->location.get_file();
      std::cout << " line " << src.pc->location.get_line();
      std::cout << std::endl << std::endl;
    }

    src = it->calling_location;
  }

  if (!thread_ended) {
    std::cout << spaces << "Next instruction to be executed:" << std::endl;
    source.prog->output_instruction(ns, "", std::cout, source.pc, true, false);
  }

  return;
}

std::vector<dstring>
goto_symex_statet::gen_stack_trace(void) const
{
  std::vector<dstring> trace;
  call_stackt::const_reverse_iterator it;
  symex_targett::sourcet src;
  int i = 0;

  // Format is a vector of strings, each recording a particular function
  // invocation.

  src = source;
  for (it = call_stack.rbegin(); it != call_stack.rend();
       it++, src = it->calling_location) {

    // Don't store current function, that can be extracted elsewhere.
    if (i++ == 0)
      continue;

    if (it->function_identifier == "") { // Top level call
      break;
    } else if (it->function_identifier == "c::main" &&
               src.pc->location == get_nil_irep()) {
      trace.push_back("<main invocation>");
    } else {
      std::string loc = it->function_identifier.as_string();
      loc += " at " + src.pc->location.get_file().as_string();
      loc += " line " + src.pc->location.get_line().as_string();
      trace.push_back(loc);
    }
  }

  return trace;
}
