/*******************************************************************\

Module: Goto Programs with Functions

Author: Daniel Kroening

Date: June 2003

\*******************************************************************/

#ifndef CPROVER_GOTO_CONVERT_FUNCTIONS_H
#define CPROVER_GOTO_CONVERT_FUNCTIONS_H

#include "goto_functions.h"
#include "goto_convert_class.h"

// just convert it all
void goto_convert(
  contextt &context,
  const optionst &options,
  goto_functionst &functions,
  message_handlert &message_handler);
  
// remove function_pointer_calls
void remove_function_pointers(
  contextt &context,
  const optionst &options,
  goto_functionst &functions,
  message_handlert &message_handler);

class goto_convert_functionst:public goto_convertt
{
public:
  void goto_convert();
  bool remove_function_pointers();
  void convert_function(const irep_idt &identifier);

  goto_convert_functionst(
    contextt &_context,
    const optionst &_options,
    goto_functionst &_functions,
    message_handlert &_message_handler);
  
  virtual ~goto_convert_functionst();

protected:
  goto_functionst &functions;
  
  static bool hide(const goto_programt &goto_program);

  //
  // function calls  
  //
  void add_return(
    goto_functionst::goto_functiont &f,
    const locationt &location);

  void remove_function_pointer(
    class value_setst &value_sets,
    goto_programt &dest,
    goto_programt::targett target);

  bool remove_function_pointers(
    class value_setst &value_sets,
    goto_programt &dest);

  bool have_function_pointers();
  bool have_function_pointers(const goto_programt &dest);
};

#endif
