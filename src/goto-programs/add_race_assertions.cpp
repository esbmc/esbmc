/*******************************************************************\

Module: Race Detection for Threaded Goto Programs

Author: Daniel Kroening
		Lucas Cordeiro (lcc08r@ecs.soton.ac.uk)

Date: February 2006
	  May 2010

\*******************************************************************/

#include <hash_cont.h>
#include <std_expr.h>
#include <expr_util.h>
#include <guard.h>

#include <pointer-analysis/value_sets.h>

#include "remove_skip.h"
#include "add_race_assertions.h"
#include "rw_set.h"

class w_guardst
{
public:
  w_guardst(contextt &_context):context(_context)
  {
  }

  std::list<irep_idt> w_guards;

  const symbolt &get_guard_symbol(const irep_idt &object)
  {
    const irep_idt identifier="tmp_"+id2string(object);

    const symbolt* s = context.find_symbol(identifier);
    if(s != nullptr)
      return *s;

    w_guards.push_back(identifier);

    symbolt new_symbol;
    new_symbol.name=identifier;
    new_symbol.base_name=identifier;
    new_symbol.type=typet("bool");
    new_symbol.static_lifetime=true;
    new_symbol.value.make_false();

    symbolt *symbol_ptr;
    context.move(new_symbol, symbol_ptr);
    return *symbol_ptr;
  }

  const exprt get_guard_symbol_expr(const irep_idt &object)
  {
    return symbol_expr(get_guard_symbol(object));
  }

  const exprt get_w_guard_expr(const rw_sett::entryt &entry)
  {
    assert(entry.w);
    return get_guard_symbol_expr(entry.object);
  }

  const exprt get_assertion(const rw_sett::entryt &entry)
  {
    return gen_not(get_guard_symbol_expr(entry.object));
  }

  void add_initialization(goto_programt &goto_program) const;


protected:
  contextt &context;
};

void w_guardst::add_initialization(goto_programt &goto_program) const
{
  goto_programt::targett t=goto_program.instructions.begin();
  const namespacet ns(context);

  for(std::list<irep_idt>::const_iterator
      it=w_guards.begin();
      it!=w_guards.end();
      it++)
  {
    exprt symbol=symbol_expr(ns.lookup(*it));
    expr2tc new_sym;
    migrate_expr(symbol, new_sym);

    t=goto_program.insert(t);
    t->type=ASSIGN;
    t->code = code_assign2tc(new_sym, gen_false_expr());

    t++;
  }
}

void add_race_assertions(
  value_setst &value_sets,
  contextt &context,
  goto_programt &goto_program,
  w_guardst &w_guards)
{
  namespacet ns(context);

  Forall_goto_program_instructions(i_it, goto_program)
  {
    goto_programt::instructiont &instruction=*i_it;

    if(instruction.is_assign())
    {
      exprt tmp_expr = migrate_expr_back(instruction.code);
      rw_sett rw_set(ns, value_sets, i_it, to_code(tmp_expr));

      if(rw_set.entries.empty()) continue;

      goto_programt::instructiont original_instruction;
      original_instruction.swap(instruction);

      instruction.make_skip();
      i_it++;

      // now add assignments for what is written -- set
      forall_rw_set_entries(e_it, rw_set)
        if(e_it->second.w)
        {
          goto_programt::targett t=goto_program.insert(i_it);

          t->type=ASSIGN;
          code_assignt theassign(
            w_guards.get_w_guard_expr(e_it->second),
            e_it->second.get_guard());

          migrate_expr(theassign, t->code);

          t->location=original_instruction.location;
          i_it=++t;
        }

      // insert original statement here
      {
        goto_programt::targett t=goto_program.insert(i_it);
        *t=original_instruction;
        i_it=++t;
      }

      // now add assignments for what is written -- reset
      forall_rw_set_entries(e_it, rw_set)
        if(e_it->second.w)
        {
          goto_programt::targett t=goto_program.insert(i_it);

          t->type=ASSIGN;
          code_assignt theassign(
            w_guards.get_w_guard_expr(e_it->second),
            false_exprt());
          migrate_expr(theassign, t->code);

          t->location=original_instruction.location;
          i_it=++t;
        }

      // now add assertion for what is read and written
      forall_rw_set_entries(e_it, rw_set)
      {
        goto_programt::targett t=goto_program.insert(i_it);

        expr2tc assert;
        migrate_expr(w_guards.get_assertion(e_it->second), assert);
        t->make_assertion(assert);
        t->location=original_instruction.location;
        t->location.comment(e_it->second.get_comment());
        i_it=++t;
      }

      i_it--; // the for loop already counts us up
    }
  }

  remove_skip(goto_program);
}

void add_race_assertions(
  value_setst &value_sets,
  contextt &context,
  goto_programt &goto_program)
{
  w_guardst w_guards(context);

  add_race_assertions(value_sets, context, goto_program, w_guards);

  w_guards.add_initialization(goto_program);
  goto_program.update();
}

void add_race_assertions(
  value_setst &value_sets,
  contextt &context,
  goto_functionst &goto_functions)
{
  w_guardst w_guards(context);

  Forall_goto_functions(f_it, goto_functions)
    add_race_assertions(value_sets, context, f_it->second.body, w_guards);

  // get "main"
  goto_functionst::function_mapt::iterator
    m_it=goto_functions.function_map.find(goto_functions.main_id());

  if(m_it!=goto_functions.function_map.end())
  {
    goto_programt &main=m_it->second.body;
    w_guards.add_initialization(main);
  }

  goto_functions.update();
}
