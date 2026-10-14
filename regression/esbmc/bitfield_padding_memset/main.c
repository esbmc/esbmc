#include <string.h>
#include <assert.h>

/* memset zeroes all 4 bytes, so the 12 padding bits above the two bitfields
   must read back as 0 through a type-punned access. gcc gives 0x000fffff. */
typedef struct
{
  int x : 12, y : 8;
} S;

int main()
{
  S s;
  memset(&s, 0, sizeof(s));
  s.x = -1;
  s.y = -1;
  unsigned v = (unsigned)*(int *)&s;
  assert((v >> 20) == 0);
  return 0;
}
