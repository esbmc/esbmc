#include <stdlib.h>
#include <assert.h>

int main(int argc, char *argv[])
{
  if(argc>1 && argv[100])
    assert(argv[100] != NULL); 
  return 0;
}
