#include <algorithms/goto_algorithms.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/remove_skip.h>

void unwind_goto_functions::unroll_loop(
  goto_programt &goto_program,
  loopst &loop)
{
  get_loop_bounds bounds(goto_program, loop);
  if(!bounds.run())
  {
    std::cout << "couldn't expand loop\n";
    return;
  }
  unsigned unwind = bounds.get_bound();
  // TODO: What happens when K == K0? Should we optimize it out?
  if(unwind == 0)
    return;

  // Get loop exit goto number
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  if(loop_exit != goto_program.instructions.begin())
  {
    goto_programt::targett t_before = loop_exit;

    if(t_before->is_goto() && is_true(t_before->guard))
    {
      // no 'fall-out'
    }
    else
    {
      // guard against 'fall-out'
      goto_programt::targett t_goto = goto_program.insert(loop_exit);

      t_goto->make_goto(loop_exit);
      t_goto->location = loop_exit->location;
      t_goto->function = loop_exit->function;
      t_goto->guard = gen_true_expr();
    }
  }
  // we make k-1 copies, to be inserted before loop_exit
  goto_programt copies;
  for(unsigned i = 1; i < unwind; i++)
  {
    // IF !COND GOTO X
    goto_programt::targett t = loop.get_original_loop_head();
    t++; // get first instruction of the loop
    for(; t != loop_exit; t++)
    {
      assert(t != goto_program.instructions.end());
      copies.add_instruction(*t);
    }
  }
  // now insert copies before loop_exit
  goto_program.destructive_insert(loop_exit, copies);
}

bool unwind_goto_functions::run()
{
  // TODO: This can be generalized into a all_loops_algorithms or similar
  Forall_goto_functions(it, goto_functions)
  {
    if(it->second.body_available)
    {
      goto_loopst goto_loops(it->first, goto_functions, it->second, this->msg);
      auto function_loops = goto_loops.get_loops();
      if(function_loops.size())
      {
        goto_functiont &goto_function = it->second;
        goto_programt &goto_program = goto_function.body;

        // Foreach loop in the function
        for(auto itt = function_loops.rbegin(); itt != function_loops.rend();
            ++itt)
        {
          unroll_loop(goto_program, *itt);
          remove_skip(goto_function.body);
        }
      }
    }
  }
  goto_functions.update();
  return true;
}

bool get_loop_bounds::run()
{
  /**
   * This looks for the following template
   * 
   * z: SYMBOL = K0
   * 
   * a: IF !(SYMBOL < K)
   * b: ... // code that only reads SYMBOL
   * b: SYMBOL++
   * 
   * If this is matched properly then set
   * bound as K - K0 and return true
   */
  goto_programt::targett t = loop.get_original_loop_head();
  symbol2tc SYMBOL;
  int K = 0;

  // 1. Check the condition. 't' should be IF !(SYMBOL < K) THEN GOTO x
  if(!t->is_goto())
    return false;
  // Pattern match SYMBOL and K, if the relation is <= then K = K + 1
  {
    auto &cond = to_not2t(t->guard).value;
    if(!is_lessthan2t(cond) && !is_lessthanequal2t(cond))
      return false;

    std::shared_ptr<relation_data> relation;
    relation = std::dynamic_pointer_cast<relation_data>(cond);
    if(relation)
    {
      if(!is_symbol2t(relation->side_1) && !is_constant_int2t(relation->side_2))
        return false;
      SYMBOL = relation->side_1;
      K = to_constant_int2t(relation->side_2).value.to_int64();
      if(K < 0)
        return false;
    }
    if(cond->expr_id == expr2t::expr_ids::lessthanequal_id)
    {
      K++;
    }
  }

  goto_programt::targett te = loop.get_original_loop_exit();

  // 2. Check for SYMBOL = SYMBOL + 1
  te--; // the previous instruction should be the increment
  if(!te->is_assign())
    return false;
  // for now only increments of one will work
  {
    auto &x = to_code_assign2t(te->code);
    if(x.target != SYMBOL || !is_add2t(x.source))
    {
      return false;
    }

    auto &add = to_add2t(x.source);
    if(add.side_1 != SYMBOL || !is_constant_int2t(add.side_2))
      return false;

    if(to_constant_int2t(add.side_2).value.to_int64() != 1)
      return false;
  }

  // 3. Look for K0
  t--; // Previous instruction from the loop creation
  int K0 = 0;
  if(!t->is_assign())
    return false;
  // Pattern matching K0 from SYMBOL = K0
  {
    auto &x = to_code_assign2t(t->code);
    if(x.target != SYMBOL || !is_constant_int2t(x.source))
    {
      return false;
    }

    K0 = to_constant_int2t(x.source).value.to_int64();
    if(K0 < 0 || K0 >= K)
      return false;
  }

  // 3. It mustn't exist an assignment over SYMBOL inside the loop
  t++;
  t++; // First instruction of the loop

  // Check if forall assignements exists one that targets SYMBOL
  for(t++; t != te; t++)
  {
    if(t->is_assign())
    {
      auto &x = to_code_assign2t(t->code);
      if(x.target == SYMBOL)
        return false;
    }
  }

  // Saves the bound
  bound = K - K0;
  return true;
}