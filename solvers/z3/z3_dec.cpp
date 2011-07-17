/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <unistd.h>
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

  if (smtlib)
    return read_z3_result();

  std::cout << "Solving with SMT Solver Z3 v" << major << "." << minor << "\n";
  //status("Solving with SMT solver Z3");

#if 0
  status(integer2string(get_number_variables_z3()) + " variables, " +
		  integer2string(get_number_vcs_z3()) + " verification conditions");
#endif

  post_process();

  return read_z3_result();
}

/*******************************************************************\

Function: z3_dect::set_encoding

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_dect::set_encoding(bool enc)
{
  set_z3_encoding(enc);
}

/*******************************************************************\

Function: z3_dect::set_smt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_dect::set_smt(bool smt)
{
  set_smtlib(smt);
  smtlib = smt;
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

Function: z3_dect::set_ecp

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_dect::set_ecp(bool ecp)
{
  set_z3_ecp(ecp);
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

Function: z3_dect::read_assert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_dect::read_assert(std::istream &in, std::string &line)
{
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

/*******************************************************************\

Function: z3_dect::decision_procedure_text

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
#if 0
std::string z3_dect::decision_procedure_text() const
{
  std::string logic;

  if (get_z3_encoding()==true)
    logic = "AUFLIRA";
  else
	logic = "QF_AUFBV";

  return "SMT "+logic+" using "+"Z3";
}
#endif
