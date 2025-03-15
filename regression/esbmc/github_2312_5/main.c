#include <assert.h>

int main(int argc, char *argv[]) 
{
  if (argc > 1) 
     assert(argv[1]);
  return 0;
}

