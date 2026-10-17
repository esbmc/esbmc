/* github #8700 */
#include <assert.h>
int main(void)
{
  signed char c = 127;
  c++;
  assert(c == -128);
  signed char d = -128;
  --d;
  assert(d == 127);
  short s = 32767;
  int old = s++;
  assert(old == 32767 && s == -32768);
  unsigned char u = 255;
  ++u;
  assert(u == 0);
  unsigned char v = 0;
  unsigned char w = v--;
  assert(w == 0 && v == 255);
  return 0;
}
