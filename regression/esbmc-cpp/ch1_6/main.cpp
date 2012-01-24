#include <iostream>
#include <cassert>

int nondet_int();

int main()
{
   int a,b,c;
   a=nondet_int();
   b=nondet_int();
   c=a+b;
   assert(c!=a+b);
   return 0; 
}


