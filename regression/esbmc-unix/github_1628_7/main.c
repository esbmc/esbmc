#include <stdio.h>
struct Foo {};
int main(void)
{
  struct Foo f;
  char buf[100];
  snprintf(buf, 100, "%s\n", f);
}
