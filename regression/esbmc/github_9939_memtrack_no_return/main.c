/* #9939: without a return, a memtrack target stays at main's closing brace. */
#include <stdlib.h>

int main(void)
{
  int *p = malloc(sizeof(int));
  p = 0;
}
