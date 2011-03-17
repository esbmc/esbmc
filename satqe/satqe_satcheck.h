/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SATQE_SATCHECK_H
#define CPROVER_SATQE_SATCHECK_H

#include <solvers/sat/satcheck.h>

#include "cube_set.h"

class satqe_satcheckt:public satcheckt
{
public:
  satqe_satcheckt();
  virtual ~satqe_satcheckt();

  void set_cube_set(cube_sett &_cube_set)
  {
    cube_set1=&_cube_set;
  }
  
  cube_sett *cube_set1;
  cube_sett *cube_set2;
  
  void quantify(literalt l)
  {
    assert(!l.sign());
    important_variables.push_back(l.var_no());
  }
  
  virtual resultt prop_solve();
  
  void set_important_variables(
    std::vector<unsigned> &_important_variables)
  {
    important_variables=_important_variables;
  }
  
  typedef satcheckt SUB;
  
protected:
  std::vector<unsigned> important_variables;
};

#endif
