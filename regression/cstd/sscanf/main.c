#include <stdio.h>
#include <assert.h>
int main(int argc, char *argv[])
{
  int life = 24;
  sscanf(argv[1],"%d", &life); 
  assert(life == 42);
  return 0;
}
