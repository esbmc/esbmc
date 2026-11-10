/* #10111 */
#include <assert.h>

int main()
{
  unsigned c$ = 0;
  for (;;)
  {
    c$ = (c$ + 1) % 3;
    assert(c$ < 3);
  }
}
