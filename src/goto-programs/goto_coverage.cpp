#include <goto-programs/goto_coverage.h>

std::string goto_coveraget::get_filename_from_path(std::string path)
{
  if (path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path;
}

void goto_coveraget::replace_all_asserts_to_guard(
  expr2tc guard,
  bool is_instrumentation)
{
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        const expr2tc old_guard = it->guard;
        if (it->is_assert())
        {
          it->guard = guard;
          if (is_instrumentation)
            it->location.property("instrumented assertion");
          else
            it->location.property("assertion");
          it->location.comment(from_expr(ns, "", old_guard));
          it->location.user_provided(true);
        }
      }
    }
}

void goto_coveraget::add_false_asserts()
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
  insert_assert(goto_program, it, guard, from_expr(ns, "", guard));
}

void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard,
  const std::string &idf)
{
  goto_programt::targett t = goto_program.insert(it);
  t->type = ASSERT;
  t->guard = guard;
  t->location = it->location;
  t->location.property("instrumented assertion");
  t->location.comment(idf);
  t->location.user_provided(true);
  it = ++t;
}

int goto_coveraget::get_total_instrument() const
{
  int total_instrument = 0;
  forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      const goto_programt &goto_program = f_it->second.body;
      forall_goto_program_instructions (it, goto_program)
      {
        if (
          it->is_assert() &&
          it->location.property().as_string() == "instrumented assertion" &&
          it->location.user_provided() == true)
        {
          total_instrument++;
        }
      }
    }
  return total_instrument;
}

// Count the total assertion instances in goto level via goto-unwind api
// run the algorithm on the copy of the original goto program
int goto_coveraget::get_total_assert_instance() const
{
  // 1. execute goto uniwnd
  bounded_loop_unroller unwind_loops;
  unwind_loops.run(goto_functions);
  // 2. calculate the number of assertion instance
  return get_total_instrument();
}

std::set<std::pair<std::string, std::string>>
goto_coveraget::get_total_cond_assert() const
{
  std::set<std::pair<std::string, std::string>> total_cond_assert = {};
  forall_goto_functions (f_it, goto_functions)
  {
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      const goto_programt &goto_program = f_it->second.body;
      forall_goto_program_instructions (it, goto_program)
      {
        if (
          it->is_assert() &&
          it->location.property().as_string() == "instrumented assertion" &&
          it->location.user_provided() == true)
        {
          std::pair<std::string, std::string> claim_pair = std::make_pair(
            it->location.comment().as_string(), it->location.as_string());
          total_cond_assert.insert(claim_pair);
        }
      }
    }
  }
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
void goto_coveraget::gen_cond_cov()
{
  // we need to skip the conditions within the built-in library
  // while kepping the file manually included by user
  // this filter, however, is unsound.. E.g. if the src filename is the same as the biuilt in library name
  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        const std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        // e.g. IF !(a > 1) THEN GOTO 3
        if (!is_true(it->guard) && it->is_goto())
        {
          // e.g.
          //    GOTO 2;
          //    2: IF(...);
          if (it->is_target())
            target_num = it->target_number;

          // preprocessing: if(true) ==> if(true == true)
          exprt guard = migrate_expr_back(it->guard);
          guard = handle_single_guard(guard);

          exprt pre_cond = nil_exprt();
          gen_cond_cov_assert(guard, pre_cond, goto_program, it);
        }
        target_num = -1;
      }
    }
}

/*
  algo:
  if(b==0 && c > 90)
  => assert(b==0)
  => assert(!(b==0));
  => assert(!(b==0 && c>90))
  => assert(!(b==0 && !(c>90)))

  if(b==0 || c > 90)
  => assert(b==0)
  => assert((b==0));
  => assert(!(!b==0 && c>90))
  => assert(!(!(b==0) && !(c>90)))
*/
void goto_coveraget::gen_cond_cov_assert(
  exprt ptr,
  exprt pre_cond,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  const auto &id = ptr.id();
  if (
    id == exprt::equality || id == exprt::notequal || id == exprt::i_lt ||
    id == exprt::i_gt || id == exprt::i_le || id == exprt::i_ge)
  {
    add_cond_cov_assert(ptr, pre_cond, goto_program, it);
  }
  else if (id == irept::id_and)
  {
    // got lhs
    gen_cond_cov_assert(ptr.op0(), pre_cond, goto_program, it);

    // update pre-condition: pre_cond && op0
    pre_cond =
      pre_cond.is_nil() ? ptr.op0() : gen_and_expr(pre_cond, ptr.op0());

    // go rhs
    gen_cond_cov_assert(ptr.op1(), pre_cond, goto_program, it);
  }
  else if (id == irept::id_or)
  {
    // got lhs
    gen_cond_cov_assert(ptr.op0(), pre_cond, goto_program, it);

    // update pre-condition: !(pre_cond && op0)
    pre_cond =
      pre_cond.is_nil() ? ptr.op0() : gen_and_expr(pre_cond, ptr.op0());
    pre_cond = gen_not_expr(pre_cond);

    // go rhs
    gen_cond_cov_assert(ptr.op1(), pre_cond, goto_program, it);
  }
  else if (id == "if")
  {
    // go left
    gen_cond_cov_assert(ptr.op0(), pre_cond, goto_program, it);

    // update pre-condition: pre_cond && op0
    exprt pre_cond_1 =
      pre_cond.is_nil() ? ptr.op0() : gen_and_expr(pre_cond, ptr.op0());

    // go mid
    gen_cond_cov_assert(ptr.op1(), pre_cond_1, goto_program, it);

    // update pre-condition: pre_cond && !op0
    exprt not_expr = gen_not_expr(ptr.op0());
    exprt pre_cond_2 =
      pre_cond.is_nil() ? not_expr : gen_and_expr(pre_cond, not_expr);

    // go right
    gen_cond_cov_assert(ptr.op2(), pre_cond_2, goto_program, it);
  }
  else
    forall_operands (op, ptr)
      gen_cond_cov_assert(*op, pre_cond, goto_program, it);
}

void goto_coveraget::add_cond_cov_assert(
  const exprt &expr,
  const exprt &pre_cond,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  expr2tc guard;
  exprt cond = pre_cond.is_nil() ? expr : gen_and_expr(pre_cond, expr);
  migrate_expr(cond, guard);

  // e.g. assert(!(a==1));  // a==1
  // the idf is used as the claim_msg
  // note that it's difference from the acutal guard.
  std::string idf = from_expr(ns, "", expr);
  make_not(guard);

  // insert assert
  insert_assert(goto_program, it, guard, idf);

  if (target_num != -1)
  {
    // update target
    std::vector<goto_programt::instructiont::targett> tgt_list;
    Forall_goto_program_instructions (itt, goto_program)
    {
      //! assume only one target
      if (
        itt->is_goto() && itt->has_target() &&
        itt->get_target()->target_number == (unsigned)target_num)
      {
        tgt_list.push_back(itt);
      }
    }

    if (!tgt_list.empty())
    {
      //! do not change the order
      // 1. rm original tgt_num
      it->target_number = -1;

      // 2. add tgt_num to the instrumentation  (x: ASSERT)
      --it;
      it->target_number = target_num;

      // 3. update src (GOTO x)
      for (auto &itt : tgt_list)
        itt->set_target(it);

      // 4. reset
      ++it;
    }
  }

  // reversal
  exprt not_expr = gen_not_expr(expr);
  cond = pre_cond.is_nil() ? not_expr : gen_and_expr(pre_cond, not_expr);
  migrate_expr(cond, guard);

  idf = from_expr(ns, "", gen_not_expr(expr));
  make_not(guard);
  insert_assert(goto_program, it, guard, idf);
}

exprt goto_coveraget::gen_no_eq_expr(const exprt &lhs, const exprt &rhs)
{
  exprt not_eq_expr = exprt("notequal", bool_type());
  exprt _lhs = lhs;
  if (lhs.type() != rhs.type())
  {
    _lhs = typecast_exprt(lhs, rhs.type());
  }
  not_eq_expr.operands().emplace_back(_lhs);
  not_eq_expr.operands().emplace_back(rhs);
  return not_eq_expr;
}

exprt goto_coveraget::gen_and_expr(const exprt &lhs, const exprt &rhs)
{
  exprt join_expr = exprt(exprt::id_and, bool_type());
  exprt _lhs = typecast_exprt(lhs, bool_type());
  exprt _rhs = typecast_exprt(rhs, bool_type());
  join_expr.operands().emplace_back(_lhs);
  join_expr.operands().emplace_back(_rhs);
  return join_expr;
}

exprt goto_coveraget::gen_not_expr(const exprt &expr)
{
  exprt not_expr = exprt(exprt::id_not, bool_type());
  not_expr.operands().emplace_back(expr);
  return not_expr;
}

/*
  This function convert single guard to a non_equal_to_false expression
  e.g. if(true) ==> if(true!=false)
*/
exprt goto_coveraget::handle_single_guard(exprt &expr)
{
  if (expr.operands().size() == 0)
  {
    exprt false_expr = false_exprt();
    return gen_no_eq_expr(expr, false_expr);
  }
  else if (expr.operands().size() == 1)
  {
    // Unary operator or typecast
    // e.g.
    //    if (!(bool)(a++)) => if(!(bool)(a++) != false)
    // note that we do not need to convert a++ to a++!=0

    if (expr.id() == exprt::typecast)
    {
      // specail handling for tenary condition
      bool has_sub_if = false;
      exprt sub = expr;
      auto op0_ptr = expr.operands().begin();
      while (sub.operands().size() == 1)
      {
        if (sub.op0().id() == "if")
        {
          has_sub_if = true;
          break;
        }
        if (!sub.has_operands())
          break;
        sub = sub.op0();
        op0_ptr = sub.operands().begin();
      }

      if (has_sub_if)
      {
        *op0_ptr = handle_single_guard(*op0_ptr);
      }
      else
      {
        exprt false_expr = false_exprt();
        return gen_no_eq_expr(expr, false_expr);
      }
    }
    else
    {
      Forall_operands (it, expr)
        *it = handle_single_guard(*it);
    }
  }
  else if (expr.operands().size() == 2)
  {
    if (expr.id() == exprt::id_and || expr.id() == exprt::id_or)
    {
      // e.g. if(a && b) ==> if(a!=0 && b!=0)
      Forall_operands (it, expr)
        *it = handle_single_guard(*it);
    }
    // "there always a typecast bool beforehand"
    // e.g. bool a[10]; if(a[1]) ==> if((bool)a[1])
    // thus we do not need to handle other 2-opd expression here
  }
  else if (expr.operands().size() == 3)
  {
    // if(a ? b:c) ==> if (a!=0 ? b!=0 : c!=0)
    expr.op0() = handle_single_guard(expr.op0());

    // special handling for function call within the guard expressions:
    // e.g.
    //    if(func() && a) ==> if(func()?a?1:0:0)
    // we do not add '!=0' to '1' and '0'.
    if (!(expr.op1().id() == irept::id_constant &&
          expr.op1().type().id() == typet::t_bool &&
          expr.op1().value().as_string() == "true"))
    {
      expr.op1() = handle_single_guard(expr.op1());
      if (expr.op1().type() != expr.type())
        expr.op1() = typecast_exprt(expr.op1(), expr.type());
    }

    if (!(expr.op2().id() == irept::id_constant &&
          expr.op2().type().id() == typet::t_bool &&
          expr.op2().value().as_string() == "false"))
    {
      expr.op2() = handle_single_guard(expr.op2());
      if (expr.op2().type() != expr.type())
        expr.op2() = typecast_exprt(expr.op2(), expr.type());
    }
  }

  // fall through
  return expr;
}
