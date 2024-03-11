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
          // preprocessing: if(true) ==> if(true == true)
          exprt guard = migrate_expr_back(it->guard);
          bool dump = false;
          guard = handle_single_guard(guard, dump);

          // split the guard
          std::list<exprt> operands;
          std::list<std::string> operators;
          dump = false;
          collect_operands(guard, operands, dump);
          collect_operators(guard, operators);
          assert(!operators.empty());

          auto opd = operands.begin();
          auto opt = operators.begin();

          // fisrt atoms
          std::set<exprt> atoms;
          collect_atom_operands(*opd, atoms);
          add_cond_cov_init_assert(*atoms.begin(), goto_program, it);

          // set up pointer to re-build the binary tree
          //   ||       <-- top_ptr
          // a    &&
          //     b   c   <--- rhs_ptr
          exprt root;
          root.operands().emplace_back(*opd);
          exprt::operandst::iterator top_ptr = root.operands().begin();
          exprt::operandst::iterator rhs_ptr = top_ptr;
          std::vector<exprt::operandst::iterator> top_ptr_stack;
          opd++;

          while (opd != operands.end() && opt != operators.end())
          {
            if (*opt == "(")
            {
              if (!top_ptr->is_empty())
              {
                // store
                top_ptr_stack.emplace_back(top_ptr);
                top_ptr = rhs_ptr;
              }
              ++opt;
              continue;
            }
            else if (*opt == ")")
            {
              if (!top_ptr_stack.empty())
              {
                // retrieve top_ptr
                top_ptr = top_ptr_stack.back();
                top_ptr_stack.pop_back();
              }
              ++opt;
              continue;
            }
            else if (*opt == "&&" || *opt == "||")
            {
              // get atom
              atoms = {};
              const exprt elem = *opd;
              collect_atom_operands(elem, atoms);
              assert(atoms.size() == 1);
              const auto &atom = *atoms.begin();
              irep_idt op = (*opt) == "&&" ? exprt::id_and : exprt::id_or;
              add_cond_cov_rhs_assert(
                op, top_ptr, rhs_ptr, root, atom, goto_program, it);
            }
            else
            {
              log_error("Unexpected operators");
              abort();
            }

            // update counter
            ++opd;
            ++opt;
          }
        }
      }
    }
}

void goto_coveraget::add_cond_cov_init_assert(
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
  const irep_idt &op_tp,
  exprt::operandst::iterator &top_ptr,
  exprt::operandst::iterator &rhs_ptr,
  const exprt &pre_cond,
  const exprt &rhs,
  goto_programt &goto_program,
  goto_programt::targett &it)
{
  // 0. store previous state
  exprt old_top = *top_ptr;

  // 1. build new joined expr
  exprt join_expr = exprt(op_tp, bool_type());
  exprt not_expr = exprt("not", bool_type());
  not_expr.operands().emplace_back(rhs);
  join_expr.operands().emplace_back(*top_ptr);
  join_expr.operands().emplace_back(not_expr);

  // 2. replace top_expr with the joined expr
  // the pre_cond is also changed during this process
  *top_ptr = join_expr;

  // 3. obtain guard
  expr2tc guard;
  migrate_expr(pre_cond.op0(), guard);
  expr2tc a_guard;
  migrate_expr(rhs, a_guard);
  make_not(a_guard);

  // 4. modified insert_assert
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

  // 5. reversal
  join_expr.clear();
  join_expr = exprt(op_tp, bool_type());
  join_expr.operands().emplace_back(old_top);
  join_expr.operands().emplace_back(rhs);
  *top_ptr = join_expr;
  migrate_expr(pre_cond.op0(), guard);
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

  // 6. update rhs_ptr
  rhs_ptr = top_ptr->operands().begin();
  rhs_ptr++;
}

/*
  - flag: if we have handled one
*/
void goto_coveraget::collect_operands(
  const exprt &expr,
  std::list<exprt> &operands,
  bool &flag)
{
  const std::string &id = expr.id().as_string();

  if ((id == "and" || id == "or") && flag == false)
  {
    bool flg0 = false;
    collect_operands(expr.op0(), operands, flg0);
    if (!flg0)
      operands.emplace_back(expr.op0());
    // operators.emplace_back(id);
    bool flg1 = false;
    collect_operands(expr.op1(), operands, flg1);
    if (!flg1)
      operands.emplace_back(expr.op1());
    flag = true;
  }
  else
  {
    forall_operands (it, expr)
      collect_operands(*it, operands, flag);
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

void goto_coveraget::collect_operators(
  const exprt &expr,
  std::list<std::string> &operators)
{
  std::string str = from_expr(expr);
  std::list<std::string> opt;
  for (std::size_t i = 0; i < str.length(); i++)
  {
    if (str[i] == '|' && str[i + 1] == '|')
    {
      opt.emplace_back("||");
      i++;
    }
    else if (str[i] == '&' && str[i + 1] == '&')
    {
      opt.emplace_back("&&");
      i++;
    }
    else if (str[i] == '(')
      opt.emplace_back("(");
    else if (str[i] == ')')
      opt.emplace_back(")");
  }

  // add implied parentheses in boolean expression
  // e.g. if(a&&b || c&&d) ==> if((a&&b) || (c&&d))
  // general rule: add parenthesis between || and &&
  // (&&||&&)
  std::list<std::string> tmp;
  std::string lst_op = "";
  std::vector<std::string> pnt_stk;
  for (auto &op : opt)
  {
    if (op == "(")
    {
      if (pnt_stk.empty())
      {
        operators.insert(operators.end(), tmp.begin(), tmp.end());
        tmp.clear();
      }
      pnt_stk.emplace_back("(");
    }
    else if (op == ")")
    {
      tmp.emplace_back(")");
      tmp.emplace_front("(");
      if (!pnt_stk.empty())
        pnt_stk.pop_back();

      if (pnt_stk.empty())
      {
        operators.insert(operators.end(), tmp.begin(), tmp.end());
        tmp.clear();
      }
    }
    else if (op == "&&" && lst_op == "||")
    {
      if (pnt_stk.empty())
      {
        operators.insert(operators.end(), tmp.begin(), tmp.end());
        tmp.clear();
      }
      pnt_stk.push_back("(");
      tmp.emplace_back(op);
    }
    else if (op == "||" && lst_op == "&&")
    {
      tmp.emplace_front("(");
      tmp.emplace_back(")");
      if (!pnt_stk.empty())
        pnt_stk.pop_back();

      if (pnt_stk.empty())
      {
        operators.insert(operators.end(), tmp.begin(), tmp.end());
        tmp.clear();
      }
      tmp.emplace_back(op);
    }
    else
    {
      tmp.emplace_back(op);
    }
    lst_op = op;
  }

  if (!tmp.empty())
  {
    while (!pnt_stk.empty())
    {
      tmp.emplace_back(")");
      tmp.emplace_front("(");
      pnt_stk.pop_back();
    }
    operators.insert(operators.end(), tmp.begin(), tmp.end());
  }
}

exprt goto_coveraget::handle_single_guard(exprt &expr, bool &flag)
{
  if (
    expr.operands().size() == 1 ||
    (expr.operands().size() == 0 && expr.id() == "constant"))
  {
    // e.g. if(!(a++)) => if(!(a++!=0) ！=0) if(true) ==> if(1==0)
    bool flg0 = false;
    Forall_operands (it, expr)
    {
      *it = handle_single_guard(*it, flg0);
    }
    flag = true;
    if (!flg0)
    {
      exprt not_eq_expr = exprt("notequal", bool_type());
      expr2tc tmp = gen_true_expr();
      exprt new_expr = migrate_expr_back(tmp);
      not_eq_expr.operands().emplace_back(expr);
      not_eq_expr.operands().emplace_back(new_expr);
      return not_eq_expr;
    }
    else
      return expr;
  }

  if (
    (expr.id() == exprt::id_and || expr.id() == exprt::id_or) && flag == false)
  {
    Forall_operands (it, expr)
    {
      bool flg0 = false;
      *it = handle_single_guard(*it, flg0);
      if (!flg0)
      {
        // e.g. (a+1) && (b+2 == 1) => (a+1 != 0) && (b+2 == 1)
        exprt not_eq_expr = exprt("notequal", bool_type());
        exprt new_expr = constant_exprt(
          integer2binary(string2integer("0"), bv_width(int_type())),
          "0",
          int_type());
        not_eq_expr.operands().emplace_back(*it);
        not_eq_expr.operands().emplace_back(new_expr);
        *it = not_eq_expr;
      }
    }
    flag = true;
  }

  const std::string &id = expr.id().as_string();
  if (!(id == "=" || id == "notequal" || id == ">" || id == "<" || id == ">=" ||
        id == "<="))
  {
    Forall_operands (it, expr)
    {
      *it = handle_single_guard(*it, flag);
    }
  }
  else
    flag = true;
  return expr;
}
