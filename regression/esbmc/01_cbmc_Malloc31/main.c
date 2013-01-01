#include <stdlib.h>
  
int
main()
{
  void *beans = malloc(0);
//  void *beans = NULL;
  assert(beans == NULL); // Should /always/ be null.
  return 0;
}
