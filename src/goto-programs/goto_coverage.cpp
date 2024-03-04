#include <goto-programs/goto_coverage.h>

// assert cov
int goto_coveraget::total_instrument = 0;
int goto_coveraget::total_assert_instance = 0;

// cond cov
std::unordered_set<std::string> goto_coveraget::total_cond_assert = {};

void goto_coveraget::make_asserts_false(goto_functionst &goto_functions)
{
  log_progress("Converting all assertions to false...");
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        const expr2tc old_guard = it->guard;
        if (it->is_assert())
        {
          it->guard = gen_false_expr();
          it->location.property("assertion");
          it->location.comment(from_expr(ns, "", old_guard));
          it->location.user_provided(true);
          total_instrument++;
        }
      }
    }
}

void goto_coveraget::make_asserts_true(goto_functionst &goto_functions)
{
  log_progress("Converting all assertions to true...");
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        const expr2tc old_guard = it->guard;
        if (it->is_assert())
        {
          it->guard = gen_true_expr();
          it->location.property("assertion");
          it->location.comment(from_expr(ns, "", old_guard));
          it->location.user_provided(true);
          total_instrument++;
        }
      }
    }
}

void goto_coveraget::add_false_asserts(goto_functionst &goto_functions)
{
  log_progress("Adding false assertions...");
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        if (it->is_end_function())
        {
          // insert an assert(0) as instrumentation BEFORE each instruction
          insert_assert(goto_program, it, gen_false_expr());
          continue;
        }

        if ((!is_true(it->guard) && it->is_goto()) || it->is_target())
        {
          it++; // add an assertion behind the instruciton
          insert_assert(goto_program, it, gen_false_expr());
          continue;
        }
      }

      goto_programt::targett it = goto_program.instructions.begin();
      insert_assert(goto_program, it, gen_false_expr());
    }
}

void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard)
{
  goto_programt::targett t = goto_program.insert(it);
  t->type = ASSERT;
  t->guard = guard;
  t->location = it->location;
  t->location.property("assertion");
  t->location.comment(from_expr(ns, "", guard));
  t->location.user_provided(true);
  it = ++t;
  total_instrument++;
}

int goto_coveraget::get_total_instrument() const
{
  return total_instrument;
}

// Count the total assertion instances in goto level via goto-unwind api
// run the algorithm on the copy of the original goto program
void goto_coveraget::count_assert_instance(goto_functionst goto_functions)
{
  // 1. execute goto uniwnd
  bounded_loop_unroller unwind_loops;
  unwind_loops.run(goto_functions);
  // 2. calculate the number of assertion instance
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        if (it->is_assert())
          total_assert_instance++;
      }
    }
}

int goto_coveraget::get_total_assert_instance() const
{
  return total_assert_instance;
}

std::unordered_set<std::string> goto_coveraget::get_total_cond_assert() const
{
  return total_cond_assert;
}

/*
  Condition Coverage: fault injection
  1. find condition statements, this includes the converted for_loop/while
  2. insert assertion instances before that statement.
  e.g.
    if (a >1)
  =>
    assert(!(a>1))
    assert(a>1)
    if(a>1)
  then run multi-property
*/
void goto_coveraget::gen_cond_cov(
  goto_functionst &goto_functions,
  const std::string &filename)
{
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        // e.g. IF !(a > 1) THEN GOTO 3
        if (
          !is_true(it->guard) && it->is_goto() &&
          filename == it->location.file().as_string())
        {
          // preprocessing
          exprt guard = migrate_expr_back(it->guard);
          guard = handle_single_guard(guard);
          log_status("111");
          // split the guard
          std::list<exprt> operands;
          std::list<irep_idt> operators;
          bool dump = false;
          collect_operands(guard, operands, operators, dump);

          assert(!operators.empty());
          for (const auto &i : operators)
            log_status("{}", i.as_string());

          for (const auto &i : operands)
            log_status("{}", i.to_string());

          auto opd = operands.begin();
          auto opt = operators.begin();

          // fisrt atoms
          std::set<exprt> atoms;
          exprt &lhs = *opd;
          collect_atom_operands(lhs, atoms);
          add_cond_cov_lhs_assert(*atoms.begin(), goto_program, it);

          opd++;
          while (opd != operands.end() && opt != operators.end())
          {
            // rhs
            atoms = {};
            const exprt &rhs = *opd;
            collect_atom_operands(rhs, atoms);
            assert(atoms.size() == 1);

            const auto &atom = *atoms.begin();
            exprt pre_cond_expr = exprt("and", bool_type());
            pre_cond_expr.operands().push_back(lhs);
            pre_cond_expr.operands().push_back(atom);
            add_cond_cov_rhs_assert(
              (*opt).as_string(), pre_cond_expr, atom, goto_program, it);
            lhs = pre_cond_expr;

            // update
            opd++;
            ++opt;
          }
        }
      }
    }
}

void goto_coveraget::add_cond_cov_lhs_assert(
  const exprt &expr,
  goto_programt &goto_program,
  goto_programt::targett &it)
{
  expr2tc guard;
  migrate_expr(expr, guard);

  insert_assert(goto_program, it, guard);
  std::string idf = from_expr(ns, "", guard) + "\t" + it->location.as_string();
  total_cond_assert.insert(idf);

  // reversal
  make_not(guard);
  insert_assert(goto_program, it, guard);
  idf = from_expr(ns, "", guard) + "\t" + it->location.as_string();
  ;
  total_cond_assert.insert(idf);
}

void goto_coveraget::add_cond_cov_rhs_assert(
  const std::string &op_tp,
  const exprt &lhs,
  const exprt &atom,
  goto_programt &goto_program,
  goto_programt::targett &it)
{
  exprt and_expr = exprt(op_tp, bool_type());
  and_expr.operands().push_back(lhs);
  and_expr.operands().push_back(atom);
  expr2tc guard;
  migrate_expr(and_expr, guard);
  expr2tc a_guard;
  migrate_expr(atom, a_guard);

  // modified insert_assert
  goto_programt::targett t = goto_program.insert(it);
  t->type = ASSERT;
  t->guard = guard;
  t->location = it->location;
  t->location.property("assertion");
  t->location.comment(from_expr(ns, "", a_guard));
  t->location.user_provided(true);
  it = ++t;
  total_instrument++;

  std::string idf =
    from_expr(ns, "", a_guard) + "\t" + it->location.as_string();
  total_cond_assert.insert(idf);

  // reversal
  and_expr.clear();
  and_expr = exprt(op_tp, bool_type());
  exprt not_expr = exprt("not", bool_type());
  not_expr.operands().push_back(atom);
  and_expr.operands().push_back(lhs);
  and_expr.operands().push_back(not_expr);
  migrate_expr(and_expr, guard);
  make_not(a_guard);
  t = goto_program.insert(it);
  t->type = ASSERT;
  t->guard = guard;
  t->location = it->location;
  t->location.property("assertion");
  t->location.comment(from_expr(ns, "", a_guard));
  t->location.user_provided(true);
  it = ++t;
  total_instrument++;

  idf = from_expr(ns, "", a_guard) + "\t" + it->location.as_string();
  total_cond_assert.insert(idf);
}

/*
  - flag: if we have handled one 
*/
void goto_coveraget::collect_operands(
  const exprt &expr,
  std::list<exprt> &operands,
  std::list<irep_idt> &operators,
  bool &flag)
{
  const std::string &id = expr.id().as_string();

  if ((id == "and" || id == "or") && flag == false)
  {
    bool flg0 = false;
    collect_operands(expr.op0(), operands, operators, flg0);
    if (!flg0)
      operands.push_back(expr.op0());
    operators.push_back(id);
    bool flg1 = false;
    collect_operands(expr.op1(), operands, operators, flg1);
    if (!flg1)
      operands.push_back(expr.op1());
    flag = true;
  }
  else
  {
    forall_operands (it, expr)
      collect_operands(*it, operands, operators, flag);
    flag |= false;
  }
}

void goto_coveraget::collect_atom_operands(
  const exprt &expr,
  std::set<exprt> &atoms)
{
  const std::string &id = expr.id().as_string();
  forall_operands (it, expr)
    collect_atom_operands(*it, atoms);
  if (
    id == "=" || id == "notequal" || id == ">" || id == "<" || id == ">=" ||
    id == "<=")
  {
    atoms.insert(expr);
  }
}

exprt goto_coveraget::handle_single_guard(exprt &expr)
{
  if (
    expr.operands().size() == 1 ||
    (expr.operands().size() == 0 && expr.id() == "constant"))
  {
    // e.g. if(!(a++)) => if(!(a++!=0) ！=0) if(true) ==> if(1==0)
    Forall_operands (it, expr)
      *it = handle_single_guard(*it);

    exprt not_eq_expr = exprt("notequal", bool_type());
    exprt new_expr = constant_exprt(
      integer2binary(string2integer("0"), bv_width(int_type())),
      "0",
      int_type());
    not_eq_expr.operands().push_back(expr);
    not_eq_expr.operands().push_back(new_expr);
    return not_eq_expr;
  }

  else if (expr.id() == exprt::id_and || expr.id() == exprt::id_or)
  {
    Forall_operands (it, expr)
    {
      std::set<exprt> tmp;
      collect_atom_operands(*it, tmp);
      if (tmp.empty())
      {
        // e.g. (a+1) && (b+2 == 1) => (a+1 != 0) && (b+2 == 1)
        exprt not_eq_expr = exprt("notequal", bool_type());
        exprt new_expr = constant_exprt(
          integer2binary(string2integer("0"), bv_width(int_type())),
          "0",
          int_type());
        not_eq_expr.operands().push_back(*it);
        not_eq_expr.operands().push_back(new_expr);
        *it = not_eq_expr;
      }
    }
  }
  else
  {
    Forall_operands (it, expr)
      *it = handle_single_guard(*it);
  }

  return expr;
}
