#include <goto-programs/remove_no_op.h>

// Determine whether the instruction is a no-op.
// Some examples of such instructions incude: a SKIP,
// "if(false) goto ...", "goto next; next: ...", (void)0, etc.
//
// \param body - goto program containing the instruction
// \param it - instruction iterator that is tested for being a no-op
// \param ignore_labels - if the caller takes care of moving labels, then even
// no-op statements carrying labels can be treated as no-op's (even though they
// may carry key information such as error labels).
// \return True, iff it is a no-op.
bool is_no_op(
  const goto_programt &body,
  goto_programt::const_targett it,
  bool ignore_labels)
{
  if (!ignore_labels && !it->labels.empty())
    return false;

  if (it->is_skip())
    return true;

  if (it->is_goto())
  {
    if (is_false(it->guard))
      return true;

    goto_programt::const_targett next_it = it;
    next_it++;

    if (next_it == body.instructions.end())
      return false;

    // A branch to the next instruction is a no-op
    // We also require the guard to be 'true'
    return is_true(it->guard) && it->has_target() &&
           it->get_target() == next_it;
  }

  if (it->is_other())
  {
    if (is_nil_expr(it->code))
      return true;

    if (is_code_skip2t(it->code))
      return true;

    if (is_code_expression2t(it->code))
    {
      const auto &code_expression = to_code_expression2t(it->code);
      const auto &expr = code_expression.operand;
      if (
        is_typecast2t(expr) && is_empty_type(expr->type) &&
        is_constant_expr(to_typecast2t(expr).from))
      {
        // something like (void)0
        return true;
      }
      else if (is_constant_expr(expr))
      {
        // something like other 0;
        return true;
      }
    }

    return false;
  }

  return false;
}

// Remove unnecessary no-op statements in a subset of the given GOTO program
//
// \param goto_program  goto program containing the instructions to be cleaned
// in the range [begin, end)
// \param begin  iterator pointing to first instruction to be considered
// \param end  iterator pointing beyond last instruction to be considered
void remove_no_op(
  goto_programt &goto_program,
  goto_programt::targett begin,
  goto_programt::targett end)
{
  // This needs to be a fixed-point, as
  // removing a no-op can turn a goto into a no-op.
  std::size_t old_size;

  do
  {
    old_size = goto_program.instructions.size();

    // maps deleted instructions to their replacement
    typedef std::map<goto_programt::targett, goto_programt::targett>
      new_targetst;
    new_targetst new_targets;

    // remove no-op statements
    for (goto_programt::instructionst::iterator it = begin; it != end;)
    {
      goto_programt::targett old_target = it;

      // for collecting labels
      std::list<irep_idt> labels;

      while (is_no_op(goto_program, it, true))
      {
        // don't remove the last no-op statement,
        // it could be a target
        if (
          it == std::prev(end) || (std::next(it)->is_end_function() &&
                                   (!labels.empty() || !it->labels.empty())))
        {
          break;
        }

        // save labels
        labels.splice(labels.end(), it->labels);
        it++;
      }

      goto_programt::targett new_target = it;

      // save labels
      it->labels.splice(it->labels.begin(), labels);

      if (new_target != old_target)
      {
        for (; old_target != new_target; ++old_target)
          new_targets[old_target] = new_target; // remember the old targets
      }
      else
        it++;
    }

    // adjust gotos across the full goto program body
    for (auto &ins : goto_program.instructions)
    {
      if (ins.is_goto() || ins.is_catch())
      {
        for (auto &target : ins.targets)
        {
          new_targetst::const_iterator result = new_targets.find(target);

          if (result != new_targets.end())
            target = result->second;
        }
      }
    }

    while (new_targets.find(begin) != new_targets.end())
      ++begin;

    // now delete the no-op's -- we do so after adjusting the
    // gotos to avoid dangling targets
    for (const auto &new_target : new_targets)
      goto_program.instructions.erase(new_target.first);

    // remove the last no-op statement unless it's a target
    goto_program.compute_target_numbers();

    if (begin != end)
    {
      goto_programt::targett last = std::prev(end);
      if (begin == last)
        ++begin;

      if (is_no_op(goto_program, last) && !last->is_target())
        goto_program.instructions.erase(last);
    }
  } while (goto_program.instructions.size() < old_size);
}

// Remove unnecessary no-op statements in the entire GOTO program
//
// \param goto_program  goto program containing the instructions to be cleaned
void remove_no_op(goto_programt &goto_program)
{
  remove_no_op(
    goto_program,
    goto_program.instructions.begin(),
    goto_program.instructions.end());

  goto_program.update();
}

// Remove unnecessary no-op statements all GOTO functions
//
// \param goto_functions  list of goto functions to be cleaned
void remove_no_op(goto_functionst &goto_functions)
{
  Forall_goto_functions (f_it, goto_functions)
    remove_no_op(
      f_it->second.body,
      f_it->second.body.instructions.begin(),
      f_it->second.body.instructions.end());

  // we may remove targets
  goto_functions.update();
}
