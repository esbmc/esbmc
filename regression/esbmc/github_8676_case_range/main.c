#include <assert.h>
int nondet_int(void);
int main(void)
{
  int v = nondet_int();
  unsigned char c = nondet_int();
  int r = 0;
  switch (v)
  {
  case -3 ... -1:
    r = 1;
  case 0:
    r += 10;
    break;
  case 7 ... 7:
    r = 3;
    break;
  case 20 ... 1000000:
    assert(v >= 20 && v <= 1000000);
    r = 4;
    break;
  default:
    assert(v < -3 || (v > 0 && v < 7) || (v > 7 && v < 20) || v > 1000000);
  }
  if (v == -2) assert(r == 11);
  if (v == 0) assert(r == 10);
  if (v == 7) assert(r == 3);
  switch (c) { case 'a' ... 'z': assert(c >= 97 && c <= 122); break; default: assert(c < 97 || c > 122); }
  unsigned u = nondet_int();
  switch (u)
  {
  case -1 ... 5:
    assert(0);
  }
  return 0;
}
