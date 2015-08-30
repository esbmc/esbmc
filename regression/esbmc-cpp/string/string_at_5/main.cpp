// string assigning

#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1, str2, str3, str4, str5;
  int i;
  str1 = string("Test string: x");
  for( i = 0 ; i < 5 ; i++ )
  	str2 = str1.at(i);

  for( i = 0 ; i < 6 ; i++ )
  	str3 = str1.at(6 - i);  
  
  for( i = 0 ; i < 7 ; i++ )
  	str4 = str1.at(7 - i);
  
  assert( (str2 == "Test ") && (str3 == "Test s") && (str4 == "Test st") );
  
  cout << str3  << endl;
  return 0;
}
