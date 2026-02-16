#ifndef CPROVER_CBMC_LOOP_IDS_H
#define CPROVER_CBMC_LOOP_IDS_H

#include <goto-programs/goto_functions.h>
#include <string>

void show_loop_numbers(const goto_functionst &goto_functions);

void show_loop_numbers(
  const goto_programt &goto_program,
  const std::string &function_name);

#endif
