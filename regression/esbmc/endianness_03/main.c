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
  /* big-endian: fst is MSB, snd is LSB */
  assert(s == 0x0102);
  return 0;
}
