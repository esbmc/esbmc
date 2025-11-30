#include <stdio.h>
union Data {
  int i;
  float f;
};
int main(void)
{
  union Data d;
  d.i = 42;
  printf("%s\n", d);
}
