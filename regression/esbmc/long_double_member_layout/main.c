#include <assert.h>
#include <stddef.h>

/* clang lays a long double out with its own width; where it is double
 * (AArch64 Darwin, Windows) that is 8 bytes, not the data model's 16. */
struct S
{
  long double ld;
  char e;
};

int main(void)
{
  struct S s;
  assert(offsetof(struct S, e) == sizeof(long double));
  assert((char *)&s.e - (char *)&s == offsetof(struct S, e));
  struct S *back = (struct S *)((char *)&s.e - offsetof(struct S, e));
  assert(back == &s);
  return 0;
}
