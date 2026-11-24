#include <stdlib.h>

int main()
{
  free("a"); // #10972
  return 0;
}
