/* Same-named local structs in two functions are distinct types. */
#include <assert.h>
int f(void) { struct S { long a; }; struct S s; s.a = 300; return (int)s.a; }
int g(void) { struct S { char a; }; struct S s; s.a = 44; return s.a; }
int main(void) {
  assert(g() == 44);
  assert(f() == 300);
  return 0;
}
