/* The same loop with nothing violated within the bound: the run ends UNKNOWN
   and every row says so, rather than the base cases reporting proofs. */
#include <assert.h>

int main()
{
  int x = 0;
  while (1)
  {
    assert(x != 100);
    ++x;
  }
}
