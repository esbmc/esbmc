/* Negative counterpart of github_5868_strto: the new definitions compute a
   real value rather than returning an unconstrained one, so a wrong claim is
   refuted instead of being satisfiable. */
#include <stdlib.h>
#include <assert.h>

int main()
{
  char *e;
  assert(strtod("2.5", &e) == 3.5);
  return 0;
}
