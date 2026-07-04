#include "shared.h"
#include <cassert>
int main()
{
  assert(use_a() == 2);
  assert(use_b() == 2);
  return 0;
}
