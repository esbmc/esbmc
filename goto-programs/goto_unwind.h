/*
 * goto_unwind.h
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#ifndef GOTO_PROGRAMS_GOTO_UNWIND_H_
#define GOTO_PROGRAMS_GOTO_UNWIND_H_

#include <std_types.h>
#include <hash_cont.h>

#include <message_stream.h>

#include "goto_functions.h"

void goto_unwind(
  goto_functionst &goto_functions,
  const namespacet &ns,
  message_handlert &message_handler);

#endif /* GOTO_PROGRAMS_GOTO_UNWIND_H_ */
