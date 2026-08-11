#include <stdio.h>

int x = 5;
int secret_key = 10;

void access_memory() 
//@requires true;
//@ensures true;
{
    int *p = &x;
    p++;//disable overflow 
    
    printf("The value of secret value is: %d\n", *p);

}

int main()
//@requires true;
//@ensures true;
{
    access_memory();

    return 0;
}
