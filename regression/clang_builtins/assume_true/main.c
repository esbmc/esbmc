#include <stdlib.h>

int foo(int x) {
   __builtin_assume(x != 0);

  if (x == 0)
    return 0;

  return 1;
}

int main(){
    foo(3);
    return 0;
}