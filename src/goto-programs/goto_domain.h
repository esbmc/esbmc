#ifndef GOTO_PROGRAMS_GOTO_DOMAIN_
#define GOTO_PROGRAMS_GOTO_DOMAIN_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_program.h>
#include <goto-programs/goto_loops.h>
#include <util/guard.h>
#include <util/message/message_stream.h>
#include <irep2/irep2_expr.h>

//helper function to neatly print vector of free vars
void print_free_vars(std::vector<expr2tc> &vars, const messaget &msg);

//gets free variable declared in the main function
//these are the candidates for domain splitting
std::vector<expr2tc>
get_free_vars_main(goto_functionst &goto_functions, const messaget &msg);

// given a goto function, inserts an assumption that splits the domain
void goto_domain_split_numeric(
  goto_programt &goto_program,
  const symbol2tc &var,
  const uint32_t val,
  const bool lt,
  const messaget &msg);

#endif
