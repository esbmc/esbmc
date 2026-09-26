#include <assert.h>

struct entry
{
  int *key;
  int *value;
  char name[128];
};

// Writing name byte by byte rewrites the struct one byte at a time. key and
// value are never touched, so each must read back as the pointer stored there
// without the formula growing with every byte written.
int main()
{
  int k = 1, v = 2;
  struct entry e;
  e.key = &k;
  e.value = &v;
  unsigned char *bytes = (unsigned char *)&e;
  for (unsigned i = 0; i < sizeof(e.name); ++i)
    bytes[2 * sizeof(int *) + i] = (unsigned char)i;
  assert(*e.key == 1 && *e.value == 2);
}
