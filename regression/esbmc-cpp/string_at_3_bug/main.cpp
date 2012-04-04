// string assigning
//TEST FAILS
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1;
  str1 = "Test string"; 
  assert(str1[2] == str1.at(3)); 		//added
  cout << str1  << endl;
  return 0;
}
