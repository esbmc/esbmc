// #10186: type 1 or 3 gives the size of s.a, so this can fail.
#include <assert.h>
struct S
{
  int a, b;
};
extern int nondet_int(void);
int main(void)
{
  struct S s;
  assert(__builtin_object_size(&s.a, nondet_int()) == 8);
}
