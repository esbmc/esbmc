#include <stdarg.h>

void test_va_copy(const char *format, ...)
{
  va_list args1, args2;
  va_start(args1, format);
  va_copy(args2, args1);
  va_end(args2);
  va_end(args1);
}

int main(void)
{
  test_va_copy("test", 42);
  return 0;
}
