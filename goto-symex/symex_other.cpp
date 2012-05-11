/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <expr_util.h>

#include "goto_symex.h"

void goto_symext::symex_other(void)
{
  const goto_programt::instructiont &instruction=*cur_state->source.pc;

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

    replace_dynamic_allocation(deref_code);
    replace_nondet(deref_code);
    dereference(deref_code, false);

    symex_cpp_delete(deref_code);
  }
  else if(statement=="free")
  {
    // ignore
  }
  else if(statement=="printf")
  {
    codet deref_code(code);

    replace_dynamic_allocation(deref_code);
    replace_nondet(deref_code);
    dereference(deref_code, false);

    expr2tc new_deref_code;
    migrate_expr(deref_code, new_deref_code);
    symex_printf(expr2tc(), new_deref_code);
  }
  else if(statement=="decl")
  {
    codet deref_code(code);

    replace_dynamic_allocation(deref_code);
    replace_nondet(deref_code);
    dereference(deref_code, false);

    if(deref_code.operands().size()==2)
      throw "two-operand decl not supported here";

    if(deref_code.operands().size()!=1)
      throw "decl expects one operand";

    if(deref_code.op0().id()!=exprt::symbol)
      throw "decl expects symbol as first operand";

    // just do the L2 renaming to preseve locality
    const irep_idt &identifier=deref_code.op0().identifier();

    std::string l1_identifier=cur_state->top().level1.get_ident_name(identifier);

    const irep_idt &original_id=
      cur_state->top().level1.get_original_name(l1_identifier);

    // increase the frame if we have seen this declaration before
    while(cur_state->top().declaration_history.find(l1_identifier)!=
          cur_state->top().declaration_history.end())
    {
      unsigned index=cur_state->top().level1.current_names[original_id];
      cur_state->top().level1.rename(original_id, index+1);
      l1_identifier=cur_state->top().level1.get_ident_name(original_id);
    }

    cur_state->top().declaration_history.insert(l1_identifier);
    cur_state->top().local_variables.insert(l1_identifier);

    // seen it before?
    // it should get a fresh value
    renaming::level2t::current_namest::iterator it=
      cur_state->level2.current_names.find(l1_identifier);

    if(it!=cur_state->level2.current_names.end())
    {
      cur_state->level2.rename(l1_identifier, it->second.count+1);
      it->second.constant = expr2tc();
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

