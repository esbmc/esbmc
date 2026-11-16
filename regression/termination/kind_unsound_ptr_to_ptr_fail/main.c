/* As kind_unsound_ptr_to_ptr, but the loop never writes **q, so the havoc
 * covers it and the inductive step still proves non-termination. */
#include <stdlib.h>

int main(void)
{
  int x = 0;
  int *p = malloc(sizeof(int));
  if (!p)
    abort();
  int **q = &p;
  *p = 0;
  while (**q < 10)
    x = 1 - x;
  free(p);
  return 0;
}
