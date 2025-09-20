#include <assert.h>
#include <stddef.h>
#include <stdint.h>

struct obj
{
  int a; // 0
  int b; // 4
  int c; // 8
};

int main()
{
  unsigned a_off = offsetof(struct obj, a);
  unsigned b_off = offsetof(struct obj, b);
  unsigned c_off = offsetof(struct obj, c);

  assert(a_off == 0);
  assert(b_off == 4);
  assert(c_off == 8);
}
