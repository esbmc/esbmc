// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
using namespace std;

int main () {

  stringstream oss;
  char val1 = 'X';
  signed char val2 = 'Y';
  unsigned char val3 = 'Z';
  
  oss << val1;
  assert(oss.str() == "X");
 
  oss << val2;
  assert(oss.str() == "XY");
  
  oss << val3;
  assert(oss.str() == "XYZ");
  
  return 0;
}
