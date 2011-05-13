/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <unistd.h>
#include <assert.h>

#include <i2string.h>
#include <str_getline.h>
#include <prefix.h>

#include <boolector.h>

#include "boolector_dec.h"

/*******************************************************************\

Function: boolector_temp_filet::boolector_temp_filet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

boolector_temp_filet::boolector_temp_filet()
{
}

/*******************************************************************\

Function: boolector_temp_filet::~boolector_temp_filet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

boolector_temp_filet::~boolector_temp_filet()
{
}

boolector_dect::boolector_dect() : boolector_convt(temp_out)
{
}

/*******************************************************************\

Function: boolector_dect::dec_solve

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

decision_proceduret::resultt boolector_dect::dec_solve()
{
  //status("Number of variables: " + integer2string(get_number_variables_boolector()));

  if (btor_lang)
    return read_boolector_result();

  post_process();
  status("Solving with SMT solver Boolector v1.4");

  return read_boolector_result();
}

/*******************************************************************\

Function: boolector_dect::read_assert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_dect::read_assert(std::istream &in, std::string &line)
{
}

/*******************************************************************\

Function: boolector_dect::read_boolector_result

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

decision_proceduret::resultt boolector_dect::read_boolector_result()
{
  int result;

  if (btor_lang)
    return D_SMTLIB;

  result = check_boolector_properties();

  if (result==BOOLECTOR_UNSAT)
	return D_UNSATISFIABLE;
  else if (result==BOOLECTOR_SAT)
	return D_SATISFIABLE;

  error("Unexpected result from Boolector");

  return D_ERROR;
}

/*******************************************************************\

Function: boolector_dect::set_file

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_dect::set_file(std::string file)
{
  set_filename(file);
}

/*******************************************************************\

Function: boolector_dect::set_smt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_dect::set_btor(bool btor)
{
  boolector_prop.btor = btor;
  btor_lang = btor;
}

