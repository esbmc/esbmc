#include <goto-programs/add_race_assertions.h>
#include <goto-programs/remove_no_op.h>
#include <goto-programs/rw_set.h>

class w_guardst
{
public:
  w_guardst(contextt &_context) : context(_context)
  {
  }

  std::list<irep_idt> w_guards;

  /// Build the RACE_CHECK(&original_expr) marker that symex replaces with the
  /// per-object guard symbol during execution.
  expr2tc get_guard_symbol_expr(const exprt &original_expr)
  {
    expr2tc operand;
    migrate_expr(original_expr, operand);
    return races_check2tc(address_of2tc(operand->type, operand));
  }

  expr2tc get_w_guard_expr(const rw_sett::entryt &entry)
  {
    return get_guard_symbol_expr(entry.original_expr);
  }

  expr2tc get_assertion(const rw_sett::entryt &entry)
  {
    return not2tc(get_guard_symbol_expr(entry.original_expr));
  }

  /// Inserts the per-object guard initializers and the global races flag at
  /// the head of @p goto_program.
  void add_initialization(goto_programt &goto_program);

protected:
  contextt &context;
};

void w_guardst::add_initialization(goto_programt &goto_program)
{
  goto_programt::targett t = goto_program.instructions.begin();
  const namespacet ns(context);

  // introduce new infinite array: __ESBMC_races_flag[]
  // initialize it to zero: ARRAY_OF(0)
  type2tc arrayt = array_type2tc(get_bool_type(), expr2tc(), true);
  const irep_idt identifier = "c:@F@__ESBMC_races_flag";
  w_guards.push_back(identifier);
  symbolt new_symbol;
  new_symbol.id = identifier;
  new_symbol.name = identifier;
  new_symbol.type = migrate_type_back(arrayt);
  new_symbol.static_lifetime = true;
  new_symbol.value.make_false();
  context.move_symbol_to_context(new_symbol);

  for (const auto &w_guard : w_guards)
  {
    const symbolt &s = *ns.lookup(w_guard);
    expr2tc symbol = symbol2tc(migrate_type(s.type), s.id);
    expr2tc falsity = s.type.is_array() ? gen_zero(symbol->type, true)
                                        : gen_false_expr();
    t = goto_program.insert(t);
    t->type = ASSIGN;
    t->code = code_assign2tc(symbol, falsity);

    t++;
  }
}

/// Builds a FUNCTION_CALL instruction invoking the void function @p name with
/// no arguments and no return value, looked up in @p context.
static expr2tc make_void_call(const contextt &context, const irep_idt &name)
{
  const symbolt &fn = *context.find_symbol(name);
  expr2tc func = symbol2tc(migrate_type(fn.type), fn.id);
  return code_function_call2tc(expr2tc(), func, std::vector<expr2tc>{});
}

void add_race_assertions(
  contextt &context,
  goto_programt &goto_program,
  w_guardst &w_guards)
{
  namespacet ns(context);

  bool is_atomic = false;

  Forall_goto_program_instructions (i_it, goto_program)
  {
    goto_programt::instructiont &instruction = *i_it;

    if (instruction.is_atomic_begin())
      is_atomic = true;

    if (
      (instruction.is_assign() || instruction.is_other() ||
       instruction.is_return() || instruction.is_goto() ||
       instruction.is_assert() || instruction.is_function_call() ||
       instruction.is_assume()) &&
      !is_atomic)
    {
      // rw_sett still consumes the legacy exprt; convert at the boundary.
      exprt tmp_expr;
      if (instruction.is_goto() || instruction.is_assert())
        tmp_expr = migrate_expr_back(instruction.guard);
      else
        tmp_expr = migrate_expr_back(instruction.code);

      rw_sett rw_set(ns, i_it, tmp_expr);

      if (rw_set.entries.empty())
        continue;

      goto_programt::instructiont original_instruction;
      original_instruction.swap(instruction);

      instruction.make_skip();
      i_it++;

      {
        goto_programt::targett t = goto_program.insert(i_it);
        t->type = FUNCTION_CALL;
        t->code = make_void_call(context, "c:@F@__ESBMC_yield");
        t->location = original_instruction.location;
        i_it = ++t;
      }

      // Avoid adding too much thread interleaving by using atomic block
      // yield();
      // atomic {Assert tmp_A == 0; tmp_A = 1; A = n;}
      // tmp_A = 0;
      // See https://github.com/esbmc/esbmc/pull/1544
      goto_programt::targett t = goto_program.insert(i_it);
      *t = ATOMIC_BEGIN;
      i_it = ++t;

      // now add assertion for what is read and written
      forall_rw_set_entries(e_it, rw_set)
      {
        goto_programt::targett t = goto_program.insert(i_it);
        t->make_assertion(w_guards.get_assertion(e_it->second));
        t->location = original_instruction.location;
        t->location.user_provided(false);
        t->location.comment(e_it->second.get_comment());
        i_it = ++t;
      }

      // now add assignments for what is written -- set
      forall_rw_set_entries(e_it, rw_set) if (e_it->second.w)
      {
        goto_programt::targett t = goto_program.insert(i_it);
        expr2tc rhs;
        migrate_expr(e_it->second.get_guard(), rhs);
        t->type = ASSIGN;
        t->code = code_assign2tc(w_guards.get_w_guard_expr(e_it->second), rhs);
        t->location = original_instruction.location;
        i_it = ++t;
      }

      // insert original statement here
      // We need to keep all instructions before the return,
      // so when we process the return we need add the
      // original instruction at the end
      if (!original_instruction.is_return() && !original_instruction.is_goto())
      {
        goto_programt::targett t = goto_program.insert(i_it);

        *t = original_instruction;
        i_it = ++t;
      }

      {
        goto_programt::targett t = goto_program.insert(i_it);

        *t = ATOMIC_END;
        i_it = ++t;
      }

      if (config.options.get_bool_option("data-races-check-only"))
      {
        goto_programt::targett t = goto_program.insert(i_it);
        t->type = FUNCTION_CALL;
        t->code = make_void_call(context, "c:@F@__ESBMC_yield");
        t->location = original_instruction.location;
        i_it = ++t;
      }

      // now add assignments for what is written -- reset
      // only write operations need to be reset:
      // tmp_A = 0;
      forall_rw_set_entries(e_it, rw_set) if (e_it->second.w)
      {
        goto_programt::targett t = goto_program.insert(i_it);
        t->type = ASSIGN;
        t->code = code_assign2tc(
          w_guards.get_w_guard_expr(e_it->second), gen_false_expr());
        t->location = original_instruction.location;
        i_it = ++t;
      }

      if (original_instruction.is_return() || original_instruction.is_goto())
      {
        goto_programt::targett t = goto_program.insert(i_it);
        *t = original_instruction;
        i_it = ++t;
      }

      i_it--; // the for loop already counts us up
    }

    if (instruction.is_atomic_end())
      is_atomic = false;
  }

  remove_no_op(goto_program);
}

void add_race_assertions(contextt &context, goto_programt &goto_program)
{
  w_guardst w_guards(context);

  add_race_assertions(context, goto_program, w_guards);

  w_guards.add_initialization(goto_program);
  goto_program.update();
}

void add_race_assertions(contextt &context, goto_functionst &goto_functions)
{
  w_guardst w_guards(context);

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->first != goto_functions.main_id())
      add_race_assertions(context, f_it->second.body, w_guards);

  // get "main"
  goto_functionst::function_mapt::iterator m_it =
    goto_functions.function_map.find(goto_functions.main_id());

  if (m_it != goto_functions.function_map.end())
  {
    goto_programt &main = m_it->second.body;
    w_guards.add_initialization(main);
  }

  goto_functions.update();
}
