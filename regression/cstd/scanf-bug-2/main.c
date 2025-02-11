#include <stdlib.h>
int main(int argc, char *argv[])
{
    int* ptr = (int*) malloc(sizeof(int));
    scanf("%12d", ptr);  // input 12345678901 overflow
    printf("%d",*ptr);   // output --539222987
}
