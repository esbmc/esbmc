/* Neither the forward condition nor the inductive step can settle this loop,
   so the k steps run out and the driver reports the table itself. */
#include <assert.h>

int main()
{
  int x = 0;
  while (1)
  {
    assert(x < 3);
    assert(x != 100);
    ++x;
  }
}
