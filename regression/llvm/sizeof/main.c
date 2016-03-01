#include <stdio.h>

int foo(int n)
{
    int vals[n];

    for (int i = 0; i < n; i++)
        vals[i] = i;
    return sizeof(vals);
}
 
int main(void)
{
  assert(foo(10) == 40);
  assert(foo(2) == 8);
}
