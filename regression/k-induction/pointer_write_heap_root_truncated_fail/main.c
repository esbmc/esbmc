// lo holds p's address truncated to an int, and symex still finds p's object
// from it. Unless the inductive step havocs that object, *p keeps its
// pre-loop 0 and the step proves *p < 3.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int *p = malloc(sizeof(int));
  *p = 0;
  int *r;
  while (nondet_int())
  {
    int lo = (int)(long)p;
    r = (int *)(long)lo;
    *r = *r + 1;
    assert(*p < 3);
  }
  return 0;
}
