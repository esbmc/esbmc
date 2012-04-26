#include <iostream>
#include <new>          // C++ standard new operator
#include <cstring>      // strcpy and strlen prototypes

int main()
{
 char first[10];
 strcpy(first, "test");
 //int tmp = strlen( first ) + 1;
 //char* firstName = new char[ tmp ];
 //strcpy( firstName, first );
 assert(strcmp(first,"test")==0);
 return 0;
}
