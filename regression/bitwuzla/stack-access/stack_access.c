// This example allows getting the value of "a" in "fun2()" only when compiled without an optimization flag

#include <stdio.h>

int *i;

void fun()
{
  int a = 13;
  i = &a;
}

void fun2()
{
  int a = 15;
}

int main()
{
  fun();
  printf("*i = %d\n", *i);
  fun2();
  printf("*i = %d\n", *i);
  return 0;
}
