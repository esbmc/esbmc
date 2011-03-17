/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include <i2string.h>
#include <str_getline.h>
#include <prefix.h>

#include "smt_dec.h"

/*******************************************************************\

Function: smt_temp_filet::smt_temp_filet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

smt_temp_filet::smt_temp_filet()
{
  temp_out_filename="smt_dec_out_"+i2string(getpid())+".tmp";

  temp_out.open(
    temp_out_filename.c_str(),
    std::ios_base::out | std::ios_base::trunc
  );

  // RB: for now statically defined
  std::string logic_str = "QF_AUFBV";

  temp_out << "(benchmark temp_call" << std::endl 
           << ":source { Automatically generated with CBMC }" << std::endl 
	   << ":status unknown" << std::endl 
	   << ":logic " << logic_str << std::endl;
}

/*******************************************************************\

Function: smt_temp_filet::~smt_temp_filet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

smt_temp_filet::~smt_temp_filet()
{
  temp_out.close();
  
  if(temp_out_filename!="")
    // RB: removed for testing
    // unlink(temp_out_filename.c_str());
    ;
    
  if(temp_result_filename!="")
    // RB: removed for testing
    // unlink(temp_result_filename.c_str());
    ;
}

/*******************************************************************\

Function: smt_dect::dec_solve

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

decision_proceduret::resultt smt_dect::dec_solve()
{
  temp_out << ":formula" << std::endl;

  // Declare definitions
  for(unsigned i=0; i<defines.size(); i++)
  {
    if(defines[i].second)
      temp_out << "(let (";
    else
      temp_out << "(flet (";

    const exprt &expr=defines[i].first;

    // Insert the identifier in the set of defines
    if(defines[i].second)
      let_id.insert(expr.op0().get_string("identifier"));
    else
      flet_id.insert(expr.op0().get_string("identifier"));

    convert_smt_expr(expr.op0());
    temp_out << " ";
    convert_smt_expr(expr.op1());
    temp_out << ")";

    temp_out << std::endl;
  }

  temp_out << "(and" << std::endl;

  for(unsigned i=0 ; i<guards.size(); i++)
  {
    literalt & l = guards[i].first;
    exprt & expr = guards[i].second;
    
    temp_out << "(iff " << smt_prop.smt_literal(l) << " ";
    convert_smt_expr(expr);
    temp_out << ")" << std::endl;
  }

  for(unsigned i=0; i<assumptions.size(); i++)
  {
    const exprt &expr=assumptions[i].first;

    if (!assumptions[i].second) smt_prop.out << "(not "; 

    smt_prop.out << " (= ";
    convert_smt_expr(expr);
    smt_prop.out << ")";

    if (!assumptions[i].second) smt_prop.out << ")"; 

    smt_prop.out << std::endl;
  }

  if(assumptions.empty() && guards.empty())
    smt_prop.out << " (not false)" << std::endl;

  // Close the lets
  for(unsigned i=0; i<defines.size(); i++)
    smt_prop.out << ")";

  // Close the "and" and the benchmark
  temp_out << "))" << std::endl;

  post_process();

  temp_out.close();

  temp_result_filename=
    "smt_dec_result_"+i2string(getpid())+".tmp";

#if 0
  std::string command = "cvc3 -lang smtlib " 
                      + temp_out_filename 
                      + " > " 
                      + temp_result_filename 
                      + " 2>&1";
#endif

#if 1
  std::string command = "yices -smt -e " 
                      + temp_out_filename 
                      + " > " 
                      + temp_result_filename 
                      + " 2>&1";
#endif

#if 0
  std::string command = "boolector --smt " 
                      + temp_out_filename 
                      + " > " 
                      + temp_result_filename 
                      + " 2>&1";
#endif
    
  system(command.c_str());

  let_id.clear();
  defines.clear();
  assumptions.clear();
  guards.clear();

  return read_smt_result();
}

/*******************************************************************\

Function: smt_dect::read_assert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_dect::read_assert(std::istream &in, std::string &line)
{
  // strip ASSERT
  line=std::string(line, strlen("(= "), std::string::npos);
  if(line=="") return;
  
  // boolean variable
  if(line[0]=='l')
  {
    // boolean
    tvt value=tvt(true);
    
    unsigned number=atoi(line.c_str()+1);
    assert(number<smt_prop.no_variables());

    line=std::string(line, strlen(" "), std::string::npos);

    if (line == std::string("false)"))
      value = tvt(false);

    smt_prop.assignment[number]=value;
  }
  else
  {
    // get identifier
    std::string::size_type pos=
      line.find(' ');

    std::string identifier=std::string(line, 1, pos-1);
    
    // get value
    if(!str_getline(in, line)) return;

    // skip spaces    
    pos=0;
    while(pos<line.size() && line[pos]==' ') pos++;
    
    // get final ")"
    std::string::size_type pos2=line.rfind(')');

    if(pos2==std::string::npos) return;    

    std::string value=std::string(line, pos, pos2-pos);
    std::cout << "> " << identifier << " <=> " << value << " <";
    std::cout << std::endl;
  }
}

/*******************************************************************\

Function: smt_dect::read_smtl_result

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

decision_proceduret::resultt smt_dect::read_smt_result()
{
  std::ifstream in(temp_result_filename.c_str());
  
  std::string line;
  
  while(str_getline(in, line))
  {
    if(has_prefix(line, "sat"))
    {
      smt_prop.reset_assignment();
    
      while(str_getline(in, line))
      {
        if(has_prefix(line, "(= "))
          read_assert(in, line);
      }
      
      return D_SATISFIABLE;
    }
    else if(has_prefix(line, "unsat"))
      return D_UNSATISFIABLE;
    else if(has_prefix(line, ""))
      continue;
  }

  error("Unexpected result from SMT-Solver");
  
  return D_ERROR;
}

