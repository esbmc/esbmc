// #10186: a nondet __builtin_object_size type selects among the modes.
#include <assert.h>
#include <stddef.h>
struct S
{
  int a, b;
};
extern int nondet_int(void);
int main(void)
{
  struct S s;
  int t = nondet_int();
  size_t n = __builtin_object_size(&s.a, t);
  assert(n == ((t == 1 || t == 3) ? sizeof(s.a) : sizeof(s)));
}
