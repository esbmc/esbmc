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
  
class goto_convert_functionst:public goto_convertt
{
public:
  void goto_convert();
  void convert_function(const irep_idt &identifier);
  void thrash_type_symbols(void);
  void rename_types(irept &type);
  void wallop_type(irep_idt name,
                   std::map<irep_idt, std::set<irep_idt> > &typenames);

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

private:
  bool inlining;

};

#endif
