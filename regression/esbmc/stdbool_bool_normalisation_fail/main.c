#include <assert.h>
#include <stdbool.h>

int main(void)
{
  unsigned flags = 0x4u;

  /* Storing 4 in a bool keeps 1, so this must be reported as violated. */
  bool found = flags & 0x4u;
  assert(found == 4);

  return 0;
}
