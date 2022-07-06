/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TYPECHECK_H
#define CPROVER_TYPECHECK_H

#include <util/expr.h>

class typecheckt
{
public:
  typecheckt()
  {
  }
  ~typecheckt() = default;

protected:
  // main function -- overload this one
  virtual void typecheck() = 0;

public:
  // call that one
  virtual bool typecheck_main();
};

#endif
