/*******************************************************************\

Module: Loop IDs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <xml.h>
#include <xml_irep.h>
#include <i2string.h>

#include "loop_analysis.h"

void show_loop_numbers(
  ui_message_handlert::uit ui,
  const goto_programt &goto_program)
{
  for(goto_programt::instructionst::const_iterator
      it=goto_program.instructions.begin();
      it!=goto_program.instructions.end();
      it++)
  {
    if(it->is_backwards_goto())
    {
      unsigned loop_id=it->loop_number;

      if(ui==ui_message_handlert::XML_UI)
      {
        xmlt xml("loop");
        xml.new_element("loop-id").data=i2string(loop_id);
        
        xmlt &l=xml.new_element();
        convert(it->location, l);
        l.name="location";
        
        std::cout << xml << std::endl;
      }
      else if(ui==ui_message_handlert::PLAIN)
      {
        std::cout << "Loop " << loop_id << ":" << std::endl;

        std::cout << "  " << it->location << std::endl;
        std::cout << std::endl;
      }
      else
        assert(false);
    }
  }
}

void show_loop_numbers(
  ui_message_handlert::uit ui,
  const goto_functionst &goto_functions)
{
  for(goto_functionst::function_mapt::const_iterator
      it=goto_functions.function_map.begin();
      it!=goto_functions.function_map.end();
      it++)
    show_loop_numbers(ui, it->second.body);
}

void mark_loop_insns(goto_programt &goto_program)
{
  struct loop_spant {
    goto_programt::instructionst::iterator start;
    goto_programt::instructionst::iterator end;
    unsigned int loop_num;
  };

  std::vector<loop_spant> loop_spans;

  for(goto_programt::instructionst::iterator
      it=goto_program.instructions.begin();
      it!=goto_program.instructions.end();
      it++) {
    if (it->is_backwards_goto()) {
      assert(it->targets.size() == 1);
      loop_spant span;
      span.start = *it->targets.begin();
      span.end = it;
      span.loop_num = it->loop_number;
      loop_spans.push_back(span);
    }
  }

  // Mark all instructions to indicate what loops they are members of.
  for (auto elem : loop_spans) {
    for (goto_programt::instructionst::iterator it = elem.start;
         it != elem.end; it++) {
      it->loop_membership.insert(elem.loop_num);
    }
  }

  // Look through the loops, and ensure that if there's any overlap between
  // them, that one loop is entirely nested within the other.
  bool well_formed = true;
  for (auto elem : loop_spans) {
    // Check each loop the first insn is in...
    for (unsigned int loopnum : elem.start->loop_membership) {
      // ..and if we're not still in that loop at the _end_ of this loop span,
      // then it's not entirely nested, and this particular program isn't well
      // formed.
      if (elem.end->loop_membership.find(loopnum) ==
          elem.end->loop_membership.end())
        well_formed = false;
    }
  }

  goto_program.loops_well_formed = well_formed;
}

void mark_loop_insns(goto_functionst &goto_functions)
{
  for(goto_functionst::function_mapt::iterator
      it=goto_functions.function_map.begin();
      it!=goto_functions.function_map.end();
      it++)
    mark_loop_insns(it->second.body);
}
