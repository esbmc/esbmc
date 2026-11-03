#include <assert.h>
#include <stddef.h>
struct C { char c; _Complex double z; };
struct F { char c; _Complex float z; };
int main(void)
{
  struct C s;
  assert(offsetof(struct C, z) == 8);
  assert((char *)&s.z - (char *)&s == 8);
  assert(sizeof(struct C) == 24);
  assert(offsetof(struct F, z) == 4);
  assert(sizeof(struct F) == 12);
  return 0;
}
