#include <cassert>
#include <stdarg.h>

int first_arg_via_copy(int count, ...)
{
  va_list args1, args2;
  va_start(args1, count);
  va_copy(args2, args1);
  int val = va_arg(args2, int);
  va_end(args2);
  va_end(args1);
  return val;
}

int main(void)
{
  int result = first_arg_via_copy(1, 42);
  assert(result == 0); // fails: result is 42
  return 0;
}
