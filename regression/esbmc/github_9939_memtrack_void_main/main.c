/* #9939: a void main has no return statement, so the target stays at the brace. */
#include <stdlib.h>
void main(void)
{
  int *p = malloc(sizeof(int));
  p = 0;
}
