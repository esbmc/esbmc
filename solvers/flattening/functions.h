/*******************************************************************\

Module: Uninterpreted Functions

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_FUNCTIONS_H
#define CPROVER_FUNCTIONS_H

#include <set>

#include "equality.h"

class functionst:public equalityt
{
public:
  functionst(propt &_prop):equalityt(_prop) { }

  void record_function_application(const exprt &index_expr);

  virtual void post_process()
  {
    add_function_constraints();
    SUB::post_process();
  }
  
  typedef equalityt SUB;
                 
protected:
  typedef std::set<exprt> applicationst;
  
  struct function_infot
  {
    applicationst applications;
  };
  
  typedef std::map<exprt, function_infot> function_mapt;
  function_mapt function_map;
  
  virtual void add_function_constraints();
  virtual void add_function_constraints(const function_infot &info);
};

#endif
