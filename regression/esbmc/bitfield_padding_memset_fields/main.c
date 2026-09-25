#include <string.h>
#include <assert.h>

/* Control for bitfield_padding_memset: the declared bitfields themselves are
   laid out correctly, so a fix must not regress the low 20 bits. */
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
  assert((v & 0xFFFFFu) == 0xFFFFFu);
  return 0;
}
