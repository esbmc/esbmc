#include <assert.h>
#include <stdbool.h>

bool t2()
{
  bool x = true;
  return x;
}

bool t1()
{
  return 1 == 2 && t2();
}

int main()
{
  int nS = true;
  int nE = t1();
  bool x = nS ? nE ?: (nS > nE) : (nS < nE);

  assert(!x);
}
