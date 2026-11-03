// A byte view of the first row of an array of byte arrays is the anchor
// itself, a fixpoint of the byte-view normalisation.
#include <assert.h>
unsigned char pool[8][32];
int main(void)
{
  int *e = (int *)&pool[0][0];
  assert((char *)e != (char *)pool[0]);
  return 0;
}
