#include <string.h>
#include <assert.h>

struct chars
{
  char fst;
  char snd;
};

int main()
{
  struct chars c = {0x01, 0x02};
  short s;
  memcpy(&s, &c, 2);
  /* little-endian: fst is LSB, snd is MSB */
  assert(s == 0x0201);
  return 0;
}
