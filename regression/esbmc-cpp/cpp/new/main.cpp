#include <iostream>
#include <new>          // C++ standard new operator
#include <cstring>      // strcpy and strlen prototypes
#include <cassert>

int main()
{
 char first[10];
 strcpy(first, "test");
 
 char* firstName = new char[ strlen( first ) + 1 ];
 strcpy( firstName, first );
 
 assert(strcmp(first,"test")==0);
 return 0;
}
