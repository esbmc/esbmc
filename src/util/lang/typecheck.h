#ifndef CPROVER_TYPECHECK_H
#define CPROVER_TYPECHECK_H

#include <util/expr.h>

class typecheckt
{
public:
  virtual ~typecheckt() = default;

  // call that one
  bool typecheck_main();

protected:
  // main function -- override this one
  virtual void typecheck() = 0;
};

#endif
