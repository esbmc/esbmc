#include <assert.h>

int defined_global = 5;
int tentative_global;        /* tentative definition, not extern */
extern int undefined_extern; /* no definition anywhere: havoc'd */

int main(void)
{
  assert(defined_global == 5);
  assert(tentative_global == 0);
  return 0;
}
