#include <stdlib.h>
#include <assert.h>

int main()
{
  // Larger than the address-space model can lay out, so the allocation must
  // fail rather than be encoded as a contradiction that proves the program.
  void *b = malloc(0xFFFFFFFFFFFFFFFCUL);
  assert(b == NULL);
  assert(0);
}
