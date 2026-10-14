#include <stdatomic.h>

/* The call is a sideeffect2t in a do-while condition. Before sideeffect2t
 * carried a location, the enclosing statement's was used and this reported
 * line 9, the `do`; the default path reports line 11, the call. §136. */
int main(void)
{
  atomic_int *p = 0;
  do
    ;
  while (atomic_load(p) < 10);
  return 0;
}
