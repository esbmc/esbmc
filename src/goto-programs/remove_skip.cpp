/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-programs/remove_skip.h>

static bool is_skip(goto_programt::instructionst::iterator it)
{
  if(it->is_skip())
    return true;

  if(it->is_goto())
  {
    if(is_false(it->guard))
      return true;

    if(it->targets.size() != 1)
      return false;

    goto_programt::instructionst::iterator next_it = it;
    next_it++;

    return it->targets.front() == next_it;
  }

  if(it->is_other())
    return is_nil_expr(it->code);

  return false;
}

void remove_skip(goto_programt &goto_program)
{
  typedef std::map<goto_programt::targett, goto_programt::targett> new_targetst;
  new_targetst new_targets;

  // remove skip statements

  for(goto_programt::instructionst::iterator it =
        goto_program.instructions.begin();
      it != goto_program.instructions.end();)
  {
    goto_programt::targett old_target = it;

    // for collecting labels
    std::list<irep_idt> labels;

    while(is_skip(it))
    {
      // don't remove the last skip statement,
      // it could be a target
      if(it == --goto_program.instructions.end())
        break;

      // save labels
      labels.splice(labels.end(), it->labels);
      it++;
    }

    goto_programt::targett new_target = it;

    // save labels
    it->labels.splice(it->labels.begin(), labels);

    if(new_target != old_target)
    {
      while(new_target != old_target)
      {
        // remember the old targets
        new_targets[old_target] = new_target;
        old_target = goto_program.instructions.erase(old_target);
      }
    }
    else
      it++;
  }

  // adjust gotos

  Forall_goto_program_instructions(i_it, goto_program)
    if(i_it->is_goto())
    {
      for(auto &target : i_it->targets)
      {
        new_targetst::const_iterator result = new_targets.find(target);

        if(result != new_targets.end())
          target = result->second;

        if(i_it == result->second)
        {
          i_it->make_assumption(gen_false_expr());
          break;
        }
      }
    }

  // remove the last skip statement unless it's a target
  if(
    !goto_program.instructions.empty() &&
    is_skip(--goto_program.instructions.end()) &&
    !goto_program.instructions.back().is_target())
    goto_program.instructions.pop_back();
}

void remove_skip(goto_functionst &goto_functions)
{
  Forall_goto_functions(f_it, goto_functions)
    remove_skip(f_it->second.body);
}
