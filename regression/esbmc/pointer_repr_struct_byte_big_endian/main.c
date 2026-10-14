#include <assert.h>

struct entry
{
  int *key;
  int *value;
  char name[16];
};

// Under --big-endian a byte offset counts from the other end of the flattened
// struct. The byte written here is in name, so key reads back unchanged.
int main()
{
  int k = 1, v = 2;
  struct entry e;
  e.key = &k;
  e.value = &v;
  unsigned char *bytes = (unsigned char *)&e;
  for (unsigned i = 0; i < 1; ++i)
    bytes[2 * sizeof(int *) + i] = 0xff;
  assert(e.key == &k);
}
