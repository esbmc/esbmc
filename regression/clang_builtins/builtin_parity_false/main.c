#include <stdio.h> 
#include <assert.h>
  
  
int main() 
{ 
    unsigned int value = 5; 
    int parity = __builtin_parity(value);
    
    assert (parity == 1);
       
    return 0; 
} 