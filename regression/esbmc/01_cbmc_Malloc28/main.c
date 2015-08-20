#include <stdio.h>
#include <stdlib.h>

int nondet_int();

unsigned int nondet_uint();
#if 0
int nondet_int(){
	
	srand(time(NULL));
 	int result = rand() % 10;
	return result;
		
}
#endif
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

main(){
	test();	
}
