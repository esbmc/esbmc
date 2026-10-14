#include <string.h>
#include <assert.h>

int main(void)
{
  /* Companion to strncmp_zero_len: strncmp on fully differing buffers with
   * n == 0 still returns 0, so the negated assertion below must be violated.
   * This pins that the SUCCESSFUL test passes because the model returns 0,
   * not because the call is optimised away. */
  assert(strncmp("abc", "xyz", 0) != 0);
  return 0;
}
