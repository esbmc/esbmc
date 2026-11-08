// The inner loop writes a heap object through the temporary that holds d
// before d++. Nothing havocs that temporary, which the inner loop assigns
// before every read, so the outer loop's inductive step traces the write to
// d and proves the property.
#include <stdlib.h>
extern void abort(void);
extern int __VERIFIER_nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  char *a = malloc(8);
  char *d = a;
  unsigned c = 0;
  while (__VERIFIER_nondet_int())
  {
    while (__VERIFIER_nondet_int())
      *d++ = 0;
    c = (c + 1) % 3;
    __VERIFIER_assert(c < 3);
  }
}
