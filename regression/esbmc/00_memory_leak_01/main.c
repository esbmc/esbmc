#include <stdlib.h>
int func(){
  char *memoryArea = malloc(10); //return_value_malloc$1
  __ESBMC_assume(memoryArea);
  char *newArea = malloc(10); //return_value_malloc$2
  __ESBMC_assume(newArea);
  memoryArea = newArea; 
  free(memoryArea);
}
void main(){
  func();
  char *test = malloc(20); //return_value_malloc$1
  __ESBMC_assume(test);
  free(test);
}
