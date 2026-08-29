/* Negative counterpart of github_5868_div: the members now carry real values
   rather than being unconstrained, so a wrong claim is refuted. */
#include <stdlib.h>
#include <assert.h>

int main()
{
  div_t d = div(7, 2);
  assert(d.quot == 4);
  return 0;
}
