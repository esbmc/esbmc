#include <goto-programs/loop_unroll.h>

bool unsound_loop_unroller::runOnLoop(loopst &loop, goto_programt &goto_program)
{
  int bound = get_loop_bounds(loop);
  if(bound <= 0)
    return false; // Can't unroll this
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
  for(unsigned i = 1; i < bound; i++)
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
  //remove_skip(goto_function.body);
  return true;
}

int bounded_loop_unroller::get_loop_bounds(loopst &loop)
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
    return -1;
  // Pattern match SYMBOL and K, if the relation is <= then K = K + 1
  {
    auto &cond = to_not2t(t->guard).value;
    if(!is_lessthan2t(cond) && !is_lessthanequal2t(cond))
      return -1;

    std::shared_ptr<relation_data> relation;
    relation = std::dynamic_pointer_cast<relation_data>(cond);
    if(relation)
    {
      if(!is_symbol2t(relation->side_1) || !is_constant_int2t(relation->side_2))
        return -1;
      SYMBOL = relation->side_1;
      K = to_constant_int2t(relation->side_2).value.to_int64();
      if(K < 0)
        return -1;
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
    return -1;
  // for now only increments of one will work
  {
    auto &x = to_code_assign2t(te->code);
    if(x.target != SYMBOL || !is_add2t(x.source))
    {
      return -1;
    }

    auto &add = to_add2t(x.source);
    if(add.side_1 != SYMBOL || !is_constant_int2t(add.side_2))
      return -1;

    if(to_constant_int2t(add.side_2).value.to_int64() != 1)
      return -1;
  }

  // 3. Look for K0
  t--; // Previous instruction from the loop creation
  int K0 = 0;
  if(!t->is_assign())
    return -1;
  // Pattern matching K0 from SYMBOL = K0
  {
    auto &x = to_code_assign2t(t->code);
    if(x.target != SYMBOL || !is_constant_int2t(x.source))
    {
      return -1;
    }

    K0 = to_constant_int2t(x.source).value.to_int64();
    if(K0 < 0 || K0 >= K)
      return -1;
  }

  // 3. It mustn't exist an assignment over SYMBOL inside the loop
  t++;
  t++; // First instruction of the loop
  // Check if forall assignements exists one that targets SYMBOL
  for(; t != te; t++)
  {
    if(t->is_assign())
    {
      auto &x = to_code_assign2t(t->code);
      if(x.target == SYMBOL)
        return -1;
    }

    else if(t->is_goto() || t->is_backwards_goto())
    {
      /* This means an inner loop or goto
       * which needs to be treated correctly
       * and every reference needs to be updated
       */
      return -1;
    }
  }
  number_of_bounded_loops++;
  return K - K0;
}