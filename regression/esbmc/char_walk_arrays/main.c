/* Byte-wise walks over an array of structs, a union and a packed struct have
 * constant trip counts. */
#include <assert.h>
#include <stdint.h>
union U { int i; char c[4]; struct { short a; short b; } s; };
struct __attribute__((packed)) P { char a; int b; char c; };
struct F { int n; char d[]; };
struct E { int v; char t; };
int main(void) {
  union U u; struct P pk; struct E arr[3];
  int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
  for (char *p = (char *)&u.c[0]; p != (char *)&u.s.b; ++p) ++n1;
  for (char *p = (char *)&pk.a; p != (char *)&pk.c; ++p) ++n2;
  for (int8_t *p = (int8_t *)&arr[0].t; p != (int8_t *)&arr[2].v; ++p) ++n3;
  for (const volatile char *p = (const volatile char *)&arr[1]; p < (const volatile char *)&arr[2].t; ++p) ++n4;
  assert(n1 == 2); assert(n2 == 5); assert(n3 == 12); assert(n4 == 12);
  return 0;
}
