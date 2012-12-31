// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
using namespace std;

int main () {

  stringstream oss;
  short val1 = 12; 
  unsigned short val2 = 99;
  
  oss << val1;
  assert(oss.str() == "12");
  
  oss << val2;
  assert(oss.str() == "1299");
  
  return 0;
}
