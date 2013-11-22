/*******************************************************************\

Module: Concrete Goto Program

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAM_H
#define CPROVER_GOTO_PROGRAM_H

#include <irep2.h>
#include <std_code.h>

#include "goto_program_template.h"

#define forall_goto_program_instructions(it, program) \
  for(goto_programt::instructionst::const_iterator it=(program).instructions.begin(); \
      it!=(program).instructions.end(); it++)

#define Forall_goto_program_instructions(it, program) \
  for(goto_programt::instructionst::iterator it=(program).instructions.begin(); \
      it!=(program).instructions.end(); it++)

#endif
