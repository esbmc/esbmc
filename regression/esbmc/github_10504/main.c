/* #10504: the call trace names functions by source name and follows the
   violated-property block. */
#include <assert.h>
int nondet_int(void);
int g;
void setup(int v) { g = v; }
void check(void) { assert(g != 3); }
int main(void)
{
  setup(nondet_int());
  check();
  return 0;
}
