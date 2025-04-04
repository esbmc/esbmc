#include <goto-programs/goto_coverage.h>

size_t goto_coveraget::total_assert = 0;
size_t goto_coveraget::total_assert_ins = 0;
std::set<std::pair<std::string, std::string>> goto_coveraget::total_cond;
size_t goto_coveraget::total_branch = 0;
size_t goto_coveraget::total_func_branch = 0;

std::string goto_coveraget::get_filename_from_path(std::string path)
{
  if (path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path;
}

/*
  replace the old_condition of all assertions
  to the new condition(guard)
*/
void goto_coveraget::replace_all_asserts_to_guard(
  const expr2tc &guard,
  bool is_instrumentation)
{
  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      // "--function" mode
      if (
        target_function != "" && !is_target_func(f_it->first, target_function))
        continue;

      goto_programt &goto_program = f_it->second.body;
      std::string cur_filename;
      Forall_goto_program_instructions (it, goto_program)
      {
        cur_filename = get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->is_assert())
          replace_assert_to_guard(guard, it, is_instrumentation);
      }
    }
}

/*
  replace the old_condition of a specific assertion
  to the new condition(guard)
*/
void goto_coveraget::replace_assert_to_guard(
  const expr2tc &guard,
  goto_programt::instructiont::targett &it,
  bool is_instrumentation)
{
  const expr2tc old_guard = it->guard;
  it->guard = guard;
  if (is_instrumentation)
    it->location.property("instrumented assertion");
  else
    it->location.property("replaced assertion");
  it->location.comment(from_expr(ns, "", old_guard));
  it->location.user_provided(true);
}

/*
Algo:
- convert all assertions to false and enable multi-property
*/
void goto_coveraget::assertion_coverage()
{
  replace_all_asserts_to_guard(gen_false_expr(), true);
  total_assert = get_total_instrument();
  total_assert_ins = get_total_assert_instance();
}

/*
Branch coverage applies to any control structure that can alter the flow of execution, including:
- if-else
- switch-case
- Loops (for, while, do-while)
- try-catch-finally (not in c)
- Early exits (return, break, continue)
The goal of branch coverage is to ensure that all possible execution paths in the program are tested.

The CBMC extends it to the entry of the function. So we will do the same.


Algo:
  1. convert assertions to true
  2. add false assertion add the beginning of the function and the branch()
*/
void goto_coveraget::branch_function_coverage()
{
  log_progress("Adding false assertions...");
  total_func_branch = 0;

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  std::unordered_set<int> catch_tgt_list;
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      // "--function" mode
      if (
        target_function != "" && !is_target_func(f_it->first, target_function))
        continue;

      goto_programt &goto_program = f_it->second.body;
      std::string cur_filename;
      bool flg = true;

      Forall_goto_program_instructions (it, goto_program)
      {
        cur_filename = get_filename_from_path(it->location.file().as_string());
        // skip if it's not the verifying files
        // probably a library
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (flg)
        {
          // add a false assert in the beginning
          // to check if the function is entered.
          insert_assert(goto_program, it, gen_false_expr());
          flg = false;
        }

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        // convert assertions to true
        if (
          it->is_assert() &&
          it->location.property().as_string() != "replaced assertion" &&
          it->location.property().as_string() != "instrumented assertion")
          replace_assert_to_guard(gen_true_expr(), it, false);

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (it->is_goto() && !is_true(it->guard))
        {
          exprt guard = migrate_expr_back(it->guard);

          if (it->is_target())
            target_num = it->target_number;
          // assert(!(a > 1));
          // assert(a > 1);
          insert_assert(goto_program, it, it->guard);
          insert_assert(goto_program, it, gen_not_expr(it->guard));
        }
      }

      flg = true;
    }

  // fix for branch coverage with kind/incr
  // It seems in kind/incr, the goto_functions used during the BMC is simplified and incomplete
  total_func_branch = get_total_instrument();

  // avoid Assertion `call_stack.back().goto_state_map.size() == 0' failed
  goto_functions.update();
}

void goto_coveraget::branch_coverage()
{
  log_progress("Adding false assertions...");
  total_branch = 0;

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  std::unordered_set<int> catch_tgt_list;
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      // "--function" mode
      if (
        target_function != "" && !is_target_func(f_it->first, target_function))
        continue;

      goto_programt &goto_program = f_it->second.body;
      std::string cur_filename;

      Forall_goto_program_instructions (it, goto_program)
      {
        cur_filename = get_filename_from_path(it->location.file().as_string());
        // skip if it's not the verifying files
        // probably a library
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        // convert assertions to true
        if (
          it->is_assert() &&
          it->location.property().as_string() != "replaced assertion" &&
          it->location.property().as_string() != "instrumented assertion")
          replace_assert_to_guard(gen_true_expr(), it, false);

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (it->is_goto() && !is_true(it->guard))
        {
          if (it->is_target())
            target_num = it->target_number;
          // assert(!(a > 1));
          // assert(a > 1);
          insert_assert(goto_program, it, it->guard);
          insert_assert(goto_program, it, gen_not_expr(it->guard));
        }
      }
    }

  total_branch = get_total_instrument();

  // avoid Assertion `call_stack.back().goto_state_map.size() == 0' failed
  goto_functions.update();
}

void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard)
{
  insert_assert(goto_program, it, guard, from_expr(ns, "", guard));
}

/*
  convert
    1: DECL x   <--- it
    ASSIGN X 1
  to
    1: ASSERT(guard);
    DECL x      <--- it
    ASSIGN X 1  
*/
void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard,
  const std::string &idf)
{
  goto_programt::instructiont instruction;
  instruction.make_assertion(guard);
  instruction.location = it->location;
  instruction.function = it->function;
  instruction.location.property("instrumented assertion");
  instruction.location.comment(idf);
  instruction.location.user_provided(true);
  goto_program.insert_swap(it++, instruction);
  it--;
}

int goto_coveraget::get_total_instrument() const
{
  int total_instrument = 0;
  forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      // speed up
      if (
        target_function != "" && !is_target_func(f_it->first, target_function))
        continue;

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
  // 1. execute goto unwind
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
      if (
        target_function != "" && !is_target_func(f_it->first, target_function))
        continue;

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
void goto_coveraget::condition_coverage()
{
  // we need to skip the conditions within the built-in library
  // while keeping the file manually included by user
  // this filter, however, is unsound.. E.g. if the src filename is the same as the builtin library name
  total_cond = {{}};

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      // "--function" mode
      if (
        target_function != "" && !is_target_func(f_it->first, target_function))
        continue;

      goto_programt &goto_program = f_it->second.body;
      std::string cur_filename;
      Forall_goto_program_instructions (it, goto_program)
      {
        cur_filename = get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        /* 
          Places that could contains condition
          1. GOTO:          if (x == 1);
          2. ASSIGN:        int x = y && z;
          3. ASSERT
          4. ASSUME
          5. FUNCTION_CALL  test((signed int)(x != y));
          6. RETURN         return x && y;
          7. Other          1?2?3:4
          The issue is that, the side-effects have been removed 
          thus the condition might have been split or modified.

          For assert, assume and goto, we know it contains GUARD
          For others, we need to convert the code back to expr and
          check there operands.
        */

        // e.g. assert(a == 1);
        if (
          it->is_assume() ||
          (it->is_assert() &&
           it->location.property().as_string() != "replaced assertion"))
        {
          auto &_guard = it->guard;
          if (!is_nil_expr(_guard))
          {
            exprt guard = migrate_expr_back(_guard);
            guard = handle_single_guard(guard, true);
            exprt pre_cond = nil_exprt();
            pre_cond.location() = it->location;
            gen_cond_cov_assert(guard, pre_cond, goto_program, it);
            // after adding the instrumentation, we convert it to constant_true
            if (!it->is_assume())
              // do not change assume, as it will modify program's logic
              replace_assert_to_guard(gen_true_expr(), it, false);
          }
        }

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (it->is_goto() && !is_true(it->guard))
        {
          // e.g.
          //    GOTO 2;
          //    2: IF(...);
          if (it->is_target())
            target_num = it->target_number;

          // preprocessing: if(true) ==> if(true == true)

          exprt guard = migrate_expr_back(it->guard);
          guard = handle_single_guard(guard, true);

          exprt pre_cond = nil_exprt();
          pre_cond.location() = it->location;
          gen_cond_cov_assert(guard, pre_cond, goto_program, it);
        }

        // e.g. bool x = (a>b);
        else if (it->is_assign())
        {
          const code_assign2t &expr = to_code_assign2t(it->code);
          const expr2tc &_rhs = expr.source;
          if (!is_nil_expr(_rhs))
          {
            exprt rhs = migrate_expr_back(_rhs);
            handle_operands_guard(rhs, goto_program, it);
          }
        }

        // a>b;
        else if (it->is_other())
        {
          if (is_code_expression2t(it->code))
          {
            const auto &code_expression = to_code_expression2t(it->code);
            const auto &_other = code_expression.operand;
            if (!is_nil_expr(_other))
            {
              exprt other = migrate_expr_back(_other);
              handle_operands_guard(other, goto_program, it);
            }
          }
        }

        // e.g. RETURN a>b;
        else if (it->is_return())
        {
          const code_return2t &code_ret = to_code_return2t(it->code);
          const auto &_ret = code_ret.operand;
          if (!is_nil_expr(_ret))
          {
            exprt ret = migrate_expr_back(_ret);
            handle_operands_guard(ret, goto_program, it);
          }
        }

        // e.g. func(a>b);
        else if (it->is_function_call())
        {
          const code_function_call2t &code_func =
            to_code_function_call2t(it->code);
          for (const expr2tc &op : code_func.operands)
          {
            if (!is_nil_expr(op))
            {
              exprt func = migrate_expr_back(op);
              handle_operands_guard(func, goto_program, it);
            }
          }
        }

        // reset target number
        target_num = -1;
      }
    }

  total_cond = get_total_cond_assert();

  // recalculate line number/ target number
  goto_functions.update();
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
  // return if we meet an atom
  if (ptr.operands().size() == 0)
    return;

  const auto &id = ptr.id();
  if (ptr.operands().size() == 1)
  {
    // (a!=0)++, !a, -a, (_Bool)(int)a
    forall_operands (op, ptr)
      gen_cond_cov_assert(*op, pre_cond, goto_program, it);
  }
  else if (ptr.operands().size() == 2)
  {
    if (
      id == exprt::equality || id == exprt::notequal || id == exprt::i_lt ||
      id == exprt::i_gt || id == exprt::i_le || id == exprt::i_ge)
    {
      forall_operands (op, ptr)
        gen_cond_cov_assert(*op, pre_cond, goto_program, it);
      add_cond_cov_assert(ptr, pre_cond, goto_program, it);
    }
    else if (id == irept::id_and)
    {
      // got lhs
      gen_cond_cov_assert(ptr.op0(), pre_cond, goto_program, it);

      // update pre-condition: pre_cond && op0
      pre_cond = pre_cond.is_nil()
                   ? ptr.op0()
                   : gen_and_expr(pre_cond, ptr.op0(), it->location);

      // go rhs
      gen_cond_cov_assert(ptr.op1(), pre_cond, goto_program, it);
    }
    else if (id == irept::id_or)
    {
      // got lhs
      gen_cond_cov_assert(ptr.op0(), pre_cond, goto_program, it);

      // update pre-condition: !(pre_cond && op0)
      pre_cond = pre_cond.is_nil()
                   ? ptr.op0()
                   : gen_and_expr(pre_cond, ptr.op0(), it->location);
      pre_cond = gen_not_expr(pre_cond, it->location);

      // go rhs
      gen_cond_cov_assert(ptr.op1(), pre_cond, goto_program, it);
    }

    else
      // a+=b; a>>(b!=0);
      forall_operands (op, ptr)
        gen_cond_cov_assert(*op, pre_cond, goto_program, it);
  }
  else if (ptr.operands().size() == 3)
  {
    // id == "if"
    // go left
    gen_cond_cov_assert(ptr.op0(), pre_cond, goto_program, it);

    // update pre-condition: pre_cond && op0
    exprt pre_cond_1 = pre_cond.is_nil()
                         ? ptr.op0()
                         : gen_and_expr(pre_cond, ptr.op0(), it->location);

    // go mid
    gen_cond_cov_assert(ptr.op1(), pre_cond_1, goto_program, it);

    // update pre-condition: pre_cond && !op0
    exprt not_expr = gen_not_expr(ptr.op0(), it->location);
    exprt pre_cond_2 = pre_cond.is_nil()
                         ? not_expr
                         : gen_and_expr(pre_cond, not_expr, it->location);

    // go right
    gen_cond_cov_assert(ptr.op2(), pre_cond_2, goto_program, it);
  }
  else
  {
    log_error("unexpected operand size");
    abort();
  }
}

void goto_coveraget::add_cond_cov_assert(
  const exprt &expr,
  const exprt &pre_cond,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  expr2tc guard;
  exprt cond =
    pre_cond.is_nil() ? expr : gen_and_expr(pre_cond, expr, it->location);
  migrate_expr(cond, guard);

  // e.g. assert(!(a==1));  // a==1
  // the idf is used as the claim_msg
  // note that it's difference from the actual guard.
  std::string idf = from_expr(ns, "", expr);
  make_not(guard);

  // insert assert
  insert_assert(goto_program, it, guard, idf);

  // reversal
  exprt not_expr = gen_not_expr(expr, it->location);
  cond = pre_cond.is_nil() ? not_expr
                           : gen_and_expr(pre_cond, not_expr, it->location);
  migrate_expr(cond, guard);

  idf = from_expr(ns, "", gen_not_expr(expr, it->location));
  make_not(guard);
  insert_assert(goto_program, it, guard, idf);
}

exprt goto_coveraget::gen_not_eq_expr(
  const exprt &lhs,
  const exprt &rhs,
  const locationt &loc)
{
  assert(loc.is_not_nil());
  exprt not_eq_expr = exprt("notequal", bool_type());
  exprt _lhs = lhs;
  if (lhs.type() != rhs.type())
  {
    _lhs = typecast_exprt(lhs, rhs.type());
    _lhs.location() = lhs.location();
  }
  not_eq_expr.operands().emplace_back(_lhs);
  not_eq_expr.operands().emplace_back(rhs);
  not_eq_expr.location() = loc;
  return not_eq_expr;
}

exprt goto_coveraget::gen_and_expr(
  const exprt &lhs,
  const exprt &rhs,
  const locationt &loc)
{
  assert(loc.is_not_nil());
  exprt join_expr = exprt(exprt::id_and, bool_type());
  exprt _lhs = lhs.type().is_bool() ? lhs : typecast_exprt(lhs, bool_type());
  _lhs.location() = lhs.location();
  exprt _rhs = rhs.type().is_bool() ? rhs : typecast_exprt(rhs, bool_type());
  _rhs.location() = rhs.location();
  join_expr.operands().emplace_back(_lhs);
  join_expr.operands().emplace_back(_rhs);
  join_expr.location() = loc;
  return join_expr;
}

exprt goto_coveraget::gen_not_expr(const exprt &expr, const locationt &loc)
{
  assert(loc.is_not_nil());
  exprt not_expr = exprt(exprt::id_not, bool_type());
  not_expr.operands().emplace_back(expr);
  not_expr.location() = loc;
  return not_expr;
}

expr2tc goto_coveraget::gen_not_expr(const expr2tc &guard)
{
  exprt _guard = migrate_expr_back(guard);
  exprt not_guard = gen_not_expr(_guard, _guard.location());
  expr2tc _guard2;
  migrate_expr(not_guard, _guard2);
  return _guard2;
}

/*
  This function convert single guard to a non_equal_to_false expression
  e.g. if(true) ==> if(true!=false)
  rule:
  1. No-op: Do nothing. This means it's a symbol or constant
  2. Binary OP: for boolean expreession, e.g. a>b, a==b, do nothing
  3. Binary OP: for and/or expresson, add on both side, if possible. Do not add if it's already a binary boolean expression in 2. 
    e.g. if(x==1 && a++) => if(x==1 && a++ !=0)
  4. Others: for any other expresison, including unary, binary and teranry, traverse its op with handle_single_guard recursivly. convert it to not equal in the top level only.
    e.g. if((bool)a+b+c) => if((bool)(a+b+c)!=0)
    typecast <--- add not equal here
    - +
      - a
      - + 
        - b
        - c
  e.g. if(a) => if(a!=0); if(true) => if(true != 0); if(a?b:c:d) => if((a?b:c:d)!=0)
  if(a==b) => if(a==b); if(a&&b) => if(a != 0 && b!=0 )
*/
exprt goto_coveraget::handle_single_guard(
  exprt &expr,
  bool top_level /* = true */)
{
  // --- Rule 1: Atomic expressions ---
  // If the expression has no operands (a symbol or constant),
  // then if it's Boolean and we're at the outer guard, wrap it with "!= false".
  if (expr.operands().empty())
  {
    if (top_level && expr.type().is_bool())
    {
      exprt false_expr = false_exprt();
      false_expr.location() = expr.location();
      return gen_not_eq_expr(expr, false_expr, expr.location());
    }
    return expr;
  }

  // --- Special-case for "not" nodes ---
  // For a "not" operator, process its operand with top_level = true so that
  // even nested atomic expressions (like x in !(!(x))) get wrapped.
  if (expr.id() == "not")
  {
    expr.op0() = handle_single_guard(expr.op0(), true);
    return expr;
  }

  // --- Helper: Recognized binary comparisons ---
  auto is_comparison = [](const exprt &e) -> bool
  {
    return (
      e.id() == exprt::equality || e.id() == exprt::notequal ||
      e.id() == exprt::i_lt || e.id() == exprt::i_gt || e.id() == exprt::i_le ||
      e.id() == exprt::i_ge);
  };

  // --- Special-case for typecasts to bool ---
  // If we have (bool)(X) and X is not already a recognized guard (comparison or logical AND/OR),
  // then unwrap the typecast and wrap X.
  if (expr.id() == exprt::typecast && expr.type().id() == typet::t_bool)
  {
    exprt inner = handle_single_guard(expr.op0(), top_level);
    if (!(is_comparison(inner) || inner.id() == exprt::id_and ||
          inner.id() == exprt::id_or))
    {
      exprt false_expr = false_exprt();
      false_expr.location() = expr.location();
      return gen_not_eq_expr(inner, false_expr, expr.location());
    }
    return inner;
  }

  // --- Process Binary Operators (exactly 2 operands) ---
  if (expr.operands().size() == 2)
  {
    // Case: Logical AND/OR operators.
    if (expr.id() == exprt::id_and || expr.id() == exprt::id_or)
    {
      // Process each operand as an independent guard (top_level = true).
      for (auto &op : expr.operands())
        op = handle_single_guard(op, true);
      // For AND/OR, we do not add extra wrapping.
      return expr;
    }
    // Case: Recognized binary comparisons.
    else if (is_comparison(expr))
    {
      // Process operands with top_level = false.
      for (auto &op : expr.operands())
        op = handle_single_guard(op, false);
      return expr;
    }
    // Case: Other binary operators (e.g. arithmetic '+').
    else
    {
      for (auto &op : expr.operands())
        op = handle_single_guard(op, false);
      if (top_level)
      {
        exprt false_expr = false_exprt();
        false_expr.location() = expr.location();
        return gen_not_eq_expr(expr, false_expr, expr.location());
      }
      return expr;
    }
  }
  else
  {
    // --- Process Non-Binary Operators (Unary, Ternary, etc.) ---
    Forall_operands (it, expr)
      *it = handle_single_guard(*it, false);

    // Special-case: if the expression is a typecast to bool, leave it unchanged.
    if (expr.id() == exprt::typecast && expr.type().id() == typet::t_bool)
      return expr;

    // For any other expression producing a Boolean value,
    // if at the outer guard (top_level true) and its id is not among our no-wrap set,
    // then wrap it with "!= false". This catches cases like member accesses.
    if (
      top_level && expr.type().is_bool() &&
      (expr.id() != exprt::id_and && expr.id() != exprt::id_or &&
       expr.id() != "not" && !is_comparison(expr)))
    {
      exprt false_expr = false_exprt();
      false_expr.location() = expr.location();
      return gen_not_eq_expr(expr, false_expr, expr.location());
    }
    return expr;
  }
}

/*
  add condition instrumentation for OTHER, ASSIGN, FUNCTION_CALL..
  whose operands might contain conditions
  we handle guards for each boolean sub-operands.
*/
void goto_coveraget::handle_operands_guard(
  exprt &expr,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  if (expr.has_operands())
  {
    auto &ops = expr.operands();
    exprt pre_cond = nil_exprt();
    pre_cond.location() = it->location;

    if (ops.size() == 1)
    {
      // e.g. RETURN ++(x&&y);
      handle_operands_guard(expr.op0(), goto_program, it);
    }
    else if (ops.size() == 2)
    {
      if (expr.id() == exprt::id_and || expr.id() == exprt::id_or)
      {
        Forall_operands (it, expr)
          // we do not need to add a !=false at top level
          // e.g. return x?1:0!= return (x?1:0)!=false
          *it = handle_single_guard(*it, false);
      }
      gen_cond_cov_assert(expr, pre_cond, goto_program, it);
    }
    else
    {
      // this could only be ternary boolean
      expr = handle_single_guard(expr, false);
      gen_cond_cov_assert(expr, pre_cond, goto_program, it);
    }
  }
}

// set the target function from "--function"
void goto_coveraget::set_target(const std::string &_tgt)
{
  target_function = _tgt;
}

// check if it's the target function
bool goto_coveraget::is_target_func(
  const irep_idt &f,
  const std::string &tgt_name) const
{
  if (ns.lookup(f) == nullptr)
  {
    log_error("Cannot find target function");
    abort();
  }

  exprt symbol = symbol_expr(*ns.lookup(f));
  if (symbol.name().as_string() != tgt_name)
    return false;

  return true;
}

// negate the condition inside the assertion
// The idea is that, if the claim is verified safe, and its negated claim is also verified safe, then we say this claim is unreachable
void goto_coveraget::negating_asserts(const std::string &tgt_fname)
{
  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      if (tgt_fname != "" && !is_target_func(f_it->first, tgt_fname))
        continue;

      goto_programt &goto_program = f_it->second.body;
      std::string cur_filename;
      Forall_goto_program_instructions (it, goto_program)
      {
        cur_filename = get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->is_assert())
        {
          expr2tc guard = it->guard;
          replace_assert_to_guard(gen_not_expr(guard), it, false);
        }
      }
    }
}