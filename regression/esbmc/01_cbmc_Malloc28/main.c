#include <stdio.h>
#include <stdlib.h>

void test() {
	
  int size = nondet_uint();
  
  if (size > 0) {
    
	int* array = (int*)(malloc(size * sizeof(int)));
        __ESBMC_assume(array);
    
	int x = size - 1;
    int y = size - 1;
    
	int index = (x + y)/2;
    
	int c = array[index];
  }
}

int main(){
	test();	
}
