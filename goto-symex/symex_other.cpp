/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <expr_util.h>
#include <rename.h>

#include "goto_symex.h"

/*******************************************************************\

Function: goto_symext::symex_other

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_other(
  const goto_functionst &goto_functions,
  statet &state,
  execution_statet &ex_state,
        unsigned node_id)
{
  const goto_programt::instructiont &instruction=*state.source.pc;

  const codet &code=to_code(instruction.code);

  const irep_idt &statement=code.get_statement();

  if(statement=="expression")
  {
    // ignore
  }
  else if(statement=="cpp_delete" ||
          statement=="cpp_delete[]")
  {
    codet deref_code(code);

    replace_dynamic_allocation(state, deref_code);
    replace_nondet(deref_code, ex_state);
    dereference(deref_code, state, false,node_id);

    symex_cpp_delete(state, deref_code);
  }
  else if(statement=="free")
  {
    // ignore
  }
  else if(statement=="printf")
  {
    codet deref_code(code);

    replace_dynamic_allocation(state, deref_code);
    replace_nondet(deref_code, ex_state);
    dereference(deref_code, state, false,node_id);

    symex_printf(state, static_cast<const exprt &>(get_nil_irep()), deref_code,node_id);
  }
  else if(statement=="decl")
  {
    codet deref_code(code);

    replace_dynamic_allocation(state, deref_code);
    replace_nondet(deref_code, ex_state);
    dereference(deref_code, state, false,node_id);

    if(deref_code.operands().size()==2)
      throw "two-operand decl not supported here";

    if(deref_code.operands().size()!=1)
      throw "decl expects one operand";

    if(deref_code.op0().id()!=exprt::symbol)
      throw "decl expects symbol as first operand";

    // just do the L2 renaming to preseve locality
    const irep_idt &identifier=deref_code.op0().identifier();

    std::string l1_identifier=state.top().level1(identifier,node_id);

    const irep_idt &original_id=
      state.top().level1.get_original_name(l1_identifier);

    // increase the frame if we have seen this declaration before
    while(state.top().declaration_history.find(l1_identifier)!=
          state.top().declaration_history.end())
    {
      unsigned index=state.top().level1.current_names[original_id];
      state.top().level1.rename(original_id, index+1,node_id);
      l1_identifier=state.top().level1(original_id,node_id);
    }

    state.top().declaration_history.insert(l1_identifier);
    state.top().local_variables.insert(l1_identifier);

    // seen it before?
    // it should get a fresh value
    statet::level2t::current_namest::iterator it=
      state.level2.current_names.find(l1_identifier);

    if(it!=state.level2.current_names.end())
    {
      state.level2.rename(l1_identifier, it->second.count+1,node_id);
      it->second.constant.make_nil();
    }
  }
  else if(statement=="nondet")
  {
    // like skip
  }
  else if(statement=="asm")
  {
    // we ignore this for now
  }
  else if (statement=="assign"){
	  assert(0);
  }
  else
    throw "goto_symext: unexpected statement: "+id2string(statement);
}

