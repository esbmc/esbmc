/* #9693: a wrapper's malloc(n) is typed by the caller's sizeof(T). */
#include <stdlib.h>
struct big
{
  char pad[9000];
  int *p;
};
struct big *g;
void *wrap(size_t n)
{
  return malloc(n);
}
int main(void)
{
  g = wrap(sizeof(struct big));
  g->p = malloc(sizeof(int));
  return 0;
}
