#include <string.h>
#include <assert.h>

/* A non-zero fill must reach the padding bits too: every one of the 4 bytes
   is 0xFF, so the type-punned read is 0xFFFFFFFF. gcc agrees. */
typedef struct
{
  int x : 12, y : 8;
} S;

int main()
{
  S s;
  memset(&s, 0xFF, sizeof(s));
  unsigned v = (unsigned)*(int *)&s;
  assert(v == 0xFFFFFFFFu);
  return 0;
}
