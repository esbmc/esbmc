// The loop writes a heap object through s, which it moves only by offsets of
// itself, so s stays in the object it entered with. The inductive step havocs
// that object and moves s anywhere inside it, and still proves the property.
#include <stdlib.h>
extern void abort(void);
extern int __VERIFIER_nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  char *a = malloc(8);
  char *s = a;
  while (__VERIFIER_nondet_int())
  {
    *s++ = 'x';
    __VERIFIER_assert(s[-1] == 'x');
  }
}
