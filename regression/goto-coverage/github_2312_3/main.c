#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


int main(int argc, char *argv[])
{
    // Ensure that at least one command-line argument is provided
    // Ensure the first argument is not NULL
    if(argc==1)
      assert(argv[1] == NULL); 

    printf("Argument 1: %s\n", argv[1]);

    return 0;
}
