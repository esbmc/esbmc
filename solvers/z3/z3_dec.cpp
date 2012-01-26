/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>

#include <i2string.h>
#include <str_getline.h>
#include <prefix.h>

#include "z3_dec.h"

/*******************************************************************\

Function: z3_temp_filet::z3_temp_filet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

z3_temp_filet::z3_temp_filet()
{
}

/*******************************************************************\

Function: z3_temp_filet::~z3_temp_filet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

z3_temp_filet::~z3_temp_filet()
{
}

/*******************************************************************\

Function: z3_dect::dec_solve

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

decision_proceduret::resultt z3_dect::dec_solve()
{

  unsigned major, minor, build, revision;
  Z3_get_version(&major, &minor, &build, &revision);

  // Add assumptions that link up literals to symbols - connections that are
  // made at a high level by prop_conv, rather than by the Z3 backend
  link_syms_to_literals();

  if (smtlib)
    return read_z3_result();

  std::cout << "Solving with SMT Solver Z3 v" << major << "." << minor << "\n";

  post_process();

  return read_z3_result();
}

/*******************************************************************\

Function: z3_dect::set_file

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_dect::set_file(std::string file)
{
  set_filename(file);
}

/*******************************************************************\

Function: z3_dect::get_unsat_core

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool z3_dect::get_unsat_core(void)
{
  return get_z3_core_size();
}

/*******************************************************************\

Function: z3_dect::get_number_of_assumptions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool z3_dect::get_number_of_assumptions(void)
{
  return get_z3_number_of_assumptions();
}

/*******************************************************************\

Function: z3_dect::set_unsat_core

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_dect::set_unsat_core(uint val)
{
  set_z3_core_size(val);
}

/*******************************************************************\

Function: z3_dect::read_z3_result

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

decision_proceduret::resultt z3_dect::read_z3_result()
{
  Z3_lbool result;

  finalize_pointer_chain();

  if (smtlib)
    return D_SMTLIB;

  result = check2_z3_properties();

  if (result==Z3_L_FALSE)
	return D_UNSATISFIABLE;
  else if (result==Z3_L_UNDEF)
	return D_UNKNOWN;
  else
	return D_SATISFIABLE;
}
