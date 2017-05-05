/*******************************************************************\
 
Module: Read goto object files.
 
Author: CM Wintersteiger
 
Date: May 2007
 
\*******************************************************************/

#ifndef READ_BIN_GOTO_OBJECT_H_
#define READ_BIN_GOTO_OBJECT_H_

#include <goto-programs/goto_functions.h>
#include <util/context.h>
#include <util/message.h>

bool read_bin_goto_object(
  std::istream &in,
  const std::string &filename,
  contextt &context,
  goto_functionst &functions,
  message_handlert &msg_hndlr);

#endif /*READ_BIN_GOTO_OBJECT_H_*/
