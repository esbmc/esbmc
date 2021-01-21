#include <stdlib.h>
#include <stdio.h>

void fun1()
{
  int *data;
  data = (int *)__builtin_alloca(sizeof(*data));
  data[0] = 0;
}
void fun2()
{
  int *data;
  data = (int *)malloc(sizeof(*data));
  if (data != (int *)0)
  {
    data[0] = 0;
  }
}
int main()
{
  printf("Running fun1()");
  fun1();
  printf("Done");
  printf("Running fun2()");
  fun2();
  printf("Done");
  return 0;
}

