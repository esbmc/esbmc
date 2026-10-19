#include <assert.h>
#include <stddef.h>

struct S
{
  int a;
  char b[8];
  int c;
};

enum
{
  v = offsetof(struct S, c)
};
_Static_assert(offsetof(struct S, c) == 12, "offset of c");

struct entry
{
  const char *name;
  size_t off;
};
static const struct entry table[] = {
  {"a", offsetof(struct S, a)},
  {"c", offsetof(struct S, c)}};

char buf[offsetof(struct S, c)];

struct bits
{
  unsigned f : offsetof(struct S, b);
};

int classify(size_t off)
{
  switch (off)
  {
  case offsetof(struct S, a):
    return 0;
  case offsetof(struct S, c):
    return 2;
  default:
    return 1;
  }
}

int main(void)
{
  assert(v == 12);
  assert(table[1].off == 12);
  assert(sizeof(buf) == 12);
  assert(classify(offsetof(struct S, c)) == 2);
  struct S s;
  char *p = (char *)&s.c - offsetof(struct S, c);
  assert((struct S *)p == &s);
  return 0;
}
