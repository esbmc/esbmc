#include "module.h"

// the typedefs are local!
typedef int t;

static t i=1;

void f()
{
  assert(i==1);
  i=3;
}

// same for structs

struct struct_tag
{
  int i;
};
