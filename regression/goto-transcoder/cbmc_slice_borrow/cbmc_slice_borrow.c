#include <stddef.h>
struct Slice { int *ptr; size_t len; };
static int sum(struct Slice s)
{
  int t = 0;
  for (size_t i = 0; i < s.len; i++) t += s.ptr[i];
  return t;
}
int main()
{
  int buf[4] = {1, 2, 3, 4};
  struct Slice whole = {buf, 4};
  struct Slice tail = {buf + 2, 2};
  __CPROVER_assert(sum(whole) == 10, "whole slice");
  __CPROVER_assert(sum(tail) == 7, "borrowed sub-slice");
  return 0;
}
