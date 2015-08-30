#include <stdlib.h>
int func(){
  char *memoryArea = malloc(10); //return_value_malloc$1
  char *newArea = malloc(10); //return_value_malloc$2
  memoryArea = newArea; 
  free(memoryArea);
}
void main(){
  func();
  char *test = malloc(20); //return_value_malloc$1
  free(test);
}
