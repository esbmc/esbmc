/* #9939: main can re-enter itself, so its last step need not be in its outer call and the target stays at the brace. */
#include <stdlib.h>
int d;
void f(void);
int main(void)
{
  if (d == 0)
  {
    d = 1;
    f();
    return 1;
  }
  int *p = malloc(sizeof(int));
  p = 0;
  return 0;
}
void f(void)
{
  main();
}
