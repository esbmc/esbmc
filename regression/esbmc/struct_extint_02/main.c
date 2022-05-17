#include <assert.h>

struct
{
  unsigned _ExtInt(9) a : 3;
  unsigned _ExtInt(9) b : 6;
  int c;
} s;

int main()
{
  s.a = 5;
  s.b = 14;
  s.c = 41023;
  assert(s.a == 5);
  assert(s.b == 14);
  assert(s.c == 41023);
}
