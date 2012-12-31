/* strcmp example */
#include <iostream>
#include <cstring>

int main ()
{
  char szKey[] = "apple";
  char szInput[80] = "appla";
  
	while (strcmp (szKey,szInput) != 0){
     std::cout << "Guess my favourite fruit? " <<
     szInput << std::endl;
		 szInput[4] = 'e';
  }
  return 0;
}
