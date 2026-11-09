/* #10111: a variable whose name contains '$' must be havocked by the
   inductive step like any other. */
#include <assert.h>

int main()
{
  unsigned c$ = 0;
  for (;;)
  {
    c$++;
    assert(c$ < 3);
  }
}
