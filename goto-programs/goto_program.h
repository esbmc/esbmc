/*******************************************************************\

Module: Concrete Goto Program

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAM_H
#define CPROVER_GOTO_PROGRAM_H

#include <irep2.h>
#include <std_code.h>

#include "goto_program_template.h"

class goto_programt:public goto_program_templatet<expr2tc, expr2tc>
{
public:
  std::ostream& output_instruction(
    const class namespacet &ns,
    const irep_idt &identifier,
    std::ostream& out,
    instructionst::const_iterator it,
    bool show_location=true,
    bool show_variables=false) const;

  goto_programt() { }  
};

#define forall_goto_program_instructions(it, program) \
  for(goto_programt::instructionst::const_iterator it=(program).instructions.begin(); \
      it!=(program).instructions.end(); it++)

#define Forall_goto_program_instructions(it, program) \
  for(goto_programt::instructionst::iterator it=(program).instructions.begin(); \
      it!=(program).instructions.end(); it++)
 
bool operator<(const goto_programt::const_targett i1,
               const goto_programt::const_targett i2);

#endif
