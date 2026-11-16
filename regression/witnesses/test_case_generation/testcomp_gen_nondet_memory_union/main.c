#include <assert.h>
#include <stddef.h>

extern void __VERIFIER_nondet_memory(void *ptr, size_t size);

union word
{
  int i;
  float f;
  char c[6];
};

int main()
{
  /* The model gives every member as a view of the same bytes; c, the widest,
   * covers them all. */
  union word w;
  __VERIFIER_nondet_memory(&w, sizeof(w));
  if (w.i == 0x01020304 && w.c[5] == 9)
    assert(0);
  return 0;
}
