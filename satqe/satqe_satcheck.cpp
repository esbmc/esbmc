/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <i2string.h>

#include "satqe_satcheck.h"

/*******************************************************************\

Function: satqe_satcheckt::satqe_satcheckt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

satqe_satcheckt::satqe_satcheckt()
{
  cube_set1=NULL;
  cube_set2=NULL;
}

/*******************************************************************\

Function: satqe_satcheckt::~satqe_satcheckt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

satqe_satcheckt::~satqe_satcheckt()
{
}

/*******************************************************************\

Function: satqe_satcheckt::prop_solve

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

propt::resultt satqe_satcheckt::prop_solve()
{
  {
    std::string msg=
      i2string(no_variables())+" variables, "+
      i2string(no_clauses())+" clauses";
    messaget::status(msg);
  }
  
  while(true)
  {  
    switch(SUB::prop_solve())
    {
    case P_SATISFIABLE:
      // SAT, record cube and add blocking clause
      {
        cubest::bitvt stars, bits;

        stars.resize(important_variables.size(), false);
        bits.resize(important_variables.size(), false);
        
        bvt c;
        c.resize(important_variables.size());
    
        for(unsigned i=0; i<important_variables.size(); i++)
        {
          unsigned v=important_variables[i];

          assert(v<no_variables());

          tvt value=l_get(literalt(v, false));

          if(value.is_true())
          {
            bits[i]=true;
            c[i]=literalt(v, true);
          }
          else if(value.is_false())
          {
            bits[i]=false;
            c[i]=literalt(v, false);
          }
          else
            assert(false);
        }

        if(cube_set1!=NULL) cube_set1->insert(stars, bits);
        if(cube_set2!=NULL) cube_set2->insert(stars, bits);

        lcnf(c);
      }
      break;
    
    case P_UNSATISFIABLE:
      // UNSAT
     return P_UNSATISFIABLE;
      
    default:
      return P_ERROR;
    }
  }
  
  return P_ERROR;
}
