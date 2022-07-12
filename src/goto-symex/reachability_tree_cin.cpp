#include <goto-symex/reachability_tree.h>
#include <iostream>

// TODO: This is the only place with istream. How to fix this?
int reachability_treet::get_ileave_direction_from_user() const
{
  std::string input;
  unsigned int tid;

  if(get_cur_state().get_active_state().guard.is_false())
    log_status("This trace's guard is false; it will not be evaluated.");

  // First of all, are there actually any valid context switch targets?
  for(tid = 0; tid < get_cur_state().threads_state.size(); tid++)
  {
    if(check_thread_viable(tid, true))
      break;
  }

  // If no threads were viable, don't present a choice.
  if(tid == get_cur_state().threads_state.size())
    return get_cur_state().threads_state.size();

  log_status("Context switch point encountered; please select a thread to run");
  log_status("Current thread states:");
  execution_states.back()->print_stack_traces(4);

  while(log_status("Input: "), std::getline(std::cin, input))
  {
    if(input == "b")
    {
      log_status("Back unimplemented");
    }
    else if(input == "q")
    {
      exit(1);
    }
    else if(input.size() <= 0)
    {
      ;
    }
    else
    {
      const char *start;
      char *end;
      start = input.c_str();
      tid = strtol(start, &end, 10);
      if(start == end)
      {
        log_status("Not a valid input");
      }
      else if(tid >= get_cur_state().threads_state.size())
      {
        log_status("Number out of range");
      }
      else
      {
        if(check_thread_viable(tid, false))
          break;
      }
    }
  }

  if(std::cin.eof())
  {
    log_status("");
    exit(1);
  }

  return tid;
}
