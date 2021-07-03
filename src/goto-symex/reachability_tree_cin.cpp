#include <goto-symex/reachability_tree.h>
#include <iostream>

// TODO: This is the only place with istream. How to fix this?
int reachability_treet::get_ileave_direction_from_user() const
{
  std::string input;
  unsigned int tid;

  if(get_cur_state().get_active_state().guard.is_false())
    message_handler.status(
      "This trace's guard is false; it will not be evaulated.");

  // First of all, are there actually any valid context switch targets?
  for(tid = 0; tid < get_cur_state().threads_state.size(); tid++)
  {
    if(check_thread_viable(tid, true))
      break;
  }

  // If no threads were viable, don't present a choice.
  if(tid == get_cur_state().threads_state.size())
    return get_cur_state().threads_state.size();

  message_handler.status(
    "Context switch point encountered; please select a thread to run");
  message_handler.status("Current thread states:");
  execution_states.back()->print_stack_traces(4);

  while(message_handler.status("Input: "), std::getline(std::cin, input))
  {
    if(input == "b")
    {
      message_handler.status("Back unimplemented");
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
        message_handler.status("Not a valid input");
      }
      else if(tid >= get_cur_state().threads_state.size())
      {
        message_handler.status("Number out of range");
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
    message_handler.status("");
    exit(1);
  }

  return tid;
}
