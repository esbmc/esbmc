// The success arm of C17 7.22.3p1 yields an object that must still be freed,
// so --memory-leak-check reports it exactly as it does without the option.
#include <stdlib.h>

int main(void)
{
  void *p = malloc(0);
  return 0;
}
