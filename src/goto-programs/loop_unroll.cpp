#include <goto-programs/loop_unroll.h>

bool unsound_loop_unroller::runOnLoop(loopst &loop, goto_programt &goto_program)
{
  int bound = get_loop_bounds(loop);
  if (bound <= 0)
    return false; // Can't unroll this

  // Get loop exit goto number
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  loop_exit->make_skip();

  /* Here the Loop goto iteraction should be added if needed
   * this is only needed if this class is combined with another
   * unsound approach. Since this class is only used for bounded
   * loops we do not need to worry. */

  // If there is an inner control flow we need a map for it
  std::map<goto_programt::targett, unsigned> target_map;
  {
    size_t count = 0;
    goto_programt::targett t = loop.get_original_loop_head();
    t->make_skip();
    t++; // get first instruction of the loop
    for (; t != loop_exit; t++, count++)
    {
      assert(
        t != goto_program.instructions.end() && "Error, got invalid loop exit");
      target_map[t] = count;
    }
  }

  // we make k-1 copies, to be inserted before loop_exit

  goto_programt copies;
  for (int i = 1; i < bound; i++)
  {
    // make a copy
    std::vector<goto_programt::targett> target_vector;
    target_vector.reserve(target_map.size());
    // IF !COND GOTO X
    goto_programt::targett t = loop.get_original_loop_head();
    t++; // get first instruction of the loop
    for (; t != loop_exit; t++)
    {
      assert(t != goto_program.instructions.end());
      goto_programt::targett copied_t = copies.add_instruction(*t);
      target_vector.push_back(copied_t);
    }

    for (unsigned i = 0; i < target_vector.size(); i++)
    {
      goto_programt::targett t = target_vector[i];
      for (auto &target : t->targets)
      {
        std::map<goto_programt::targett, unsigned>::const_iterator m_it =
          target_map.find(target);
        if (m_it != target_map.end()) // intra-loop?
        {
          assert(m_it->second < target_vector.size());
          target = target_vector[m_it->second];
        }
      }
    }
  }
  // now insert copies before loop_exit
  goto_program.destructive_insert(loop_exit, copies);
  return true;
}

int bounded_loop_unroller::get_loop_bounds(loopst &loop)
{
  /**
   * This looks for the following template
   *
   * z: symbol = k0
   *
   * a: IF !(symbol < k) GOTO [...]
   * b: ... // code that only reads symbol
   * b: symbol++
   *
   * If this is matched properly then set
   * bound as k - k0 and return true
   */
  goto_programt::targett t = loop.get_original_loop_head();
  expr2tc symbol;
  int k = 0;

  // 1. Check the condition. 't' should be IF !(symbol < K) THEN GOTO x
  if (!t->is_goto() || !is_not2t(t->guard))
    return -1;

  // Pattern match symbol and K, if the relation is <= then K = K + 1
  {
    auto &cond = to_not2t(t->guard).value;
    if (const expr2tc cond_simp = cond->do_simplify(); !is_nil_expr(cond_simp))
      cond = cond_simp;
    if (!is_lessthan2t(cond) && !is_lessthanequal2t(cond))
      return -1;

    if (
      !is_symbol2t(*cond->get_sub_expr(0)) ||
      !is_constant_int2t(*cond->get_sub_expr(1)))
      return -1;

    symbol = *cond->get_sub_expr(0);
    k = to_constant_int2t(*cond->get_sub_expr(1)).value.to_int64();
    if (k < 0)
      return -1;

    if (is_lessthanequal2t(cond))
      k++;
  }

  goto_programt::targett te = loop.get_original_loop_exit();

  // 2. Check for symbol = symbol + 1
  te--; // the previous instruction should be the increment
  if (!te->is_assign())
    return -1;
  // for now only increments of one will work
  {
    auto &x = to_code_assign2t(te->code);
    if (x.target != symbol || !is_add2t(x.source))
      return -1;

    auto &add = to_add2t(x.source);
    if (add.side_1 != symbol || !is_constant_int2t(add.side_2))
      return -1;

    if (to_constant_int2t(add.side_2).value.to_int64() != 1)
      return -1;
  }

  // 3. Look for k0
  t--; // Previous instruction from the loop creation
  int k0 = 0;
  if (!t->is_assign())
    return -1;
  // Pattern matching k0 from symbol = k0
  {
    auto &x = to_code_assign2t(t->code);
    if (x.target != symbol || !is_constant_int2t(x.source))
      return -1;

    k0 = to_constant_int2t(x.source).value.to_int64();
    if (k0 < 0 || k0 >= k)
      return -1;
  }

  // 3. It mustn't exist an assignment over symbol inside the loop
  t++;
  t++; // First instruction of the loop
  // Check if forall assignments exists one that targets symbol
  for (; t != te; t++)
    if (t->is_assign())
    {
      auto &x = to_code_assign2t(t->code);
      if (x.target == symbol)
        return -1;
    }

  int bound = k - k0;
  if (bound <= 0 || (size_t)bound > unroll_limit)
    return 0;

  number_of_bounded_loops++;
  return bound;
}
