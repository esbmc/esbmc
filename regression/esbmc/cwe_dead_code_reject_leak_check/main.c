// --dead-code-check forces a SUCCESSFUL verdict, so a genuine SAT violation
// from a symex-injected safety check (memory-leak here, also deadlock/race)
// would be silently masked. Reject the combination so a real leak is never
// hidden behind a dead-code run (issue #4495).
#include <stdlib.h>
int main(void)
{
  int *p = malloc(sizeof(int));
  return p ? 0 : 1;
}
