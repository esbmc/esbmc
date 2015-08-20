#include "module.h"

typedef short int t;

t i=2;

int main()
{
  assert(i==2);
  
  f();
  
  assert(i==2);
}

struct struct_tag
{
  short int i;
};
