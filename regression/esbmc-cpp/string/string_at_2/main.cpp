// string assigning

//Case test operator


#include <iostream>
#include <string>
#include <cassert>

using namespace std;

int main ()
{
  string str1;
  str1 = string("Test string");
  char str2 = 'x';
  for(int i = 0;i < 11;i++){
  	if(str1.at(i) == 's')
  		str1[i] = str2;
  }
  assert(str1 == "Text xtring");
  
  return 0;
}
