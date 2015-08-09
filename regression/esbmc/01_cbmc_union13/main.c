// Test that union literal flattening works as advertised.

#include <stdbool.h>

int test;

union foo {
  struct {
    bool a;
    int b;
    short c;
    unsigned long d;
    bool dd;
    int *e;
    char ee;
    struct {
      short f;
      int g;
    } h;
    char i;
    unsigned long j;
  } xyzzy;
  char youwhat;
} bar = { true, 1234, 5555, 0x123456789abcdef, false, &test, true, { 3333, 0x5a5a5a5a }, 3, 0xfedcba987654321 } ;

int
main()
{
  assert(bar.xyzzy.a == true);
  assert(bar.xyzzy.b == 1234);
  assert(bar.xyzzy.c == 5555);
  assert(bar.xyzzy.d == 0x123456789abcdef);
  assert(bar.xyzzy.dd == false);
  assert(bar.xyzzy.e == &test);
  assert(bar.xyzzy.ee == true);
  assert(bar.xyzzy.h.f == 3333);
  assert(bar.xyzzy.h.g == 0x5a5a5a5a);
  assert(bar.xyzzy.i == 3);
  assert(bar.xyzzy.j == 0xfedcba987654321);
  return 0;
}
