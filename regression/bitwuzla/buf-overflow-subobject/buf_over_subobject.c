#include <stdio.h>

#define LEN 6

struct my_type
{
  int array[LEN];
  int num;
} var;

int main()
{
  var.num = 999999;
  printf("var.num = %d\n", var.num);
  int *ptr = &var.array;
  for(size_t i = 0; i < LEN + 1; i++)
    ptr++;

  *ptr = 666666;
  printf("var.num = %d\n", var.num);

  return 0;
}
