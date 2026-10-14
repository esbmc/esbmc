/* Same-named local structs in two functions are distinct types. */
#include <assert.h>
#include <string.h>
int f(void) { struct S { int a; int b; } s; memset(&s, 0, sizeof s); *(int *)&s = 7; return s.a; }
int g(void) { struct S { int b; int a; } s; memset(&s, 0, sizeof s); *(int *)&s = 7; return s.a; }
int main(void) {
  assert(f() == 7);
  assert(g() == 7);
  return 0;
}
