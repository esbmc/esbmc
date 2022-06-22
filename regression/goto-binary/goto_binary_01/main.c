// esbmc  --output-goto main.goto

#include<stdio.h>

int main()
{
    int x = 1;
    if (x < 2)
        x = 2;
    return x;
}
