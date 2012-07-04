// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstring>
using namespace std;

int main () {

  stringstream oss;
  char* val1 = new char[256];
  signed char* val2;
  unsigned char* val3;
  strcpy(val1, "test");
  *val2 = 'A';
  *val3 = 'B';
  
  oss << val1;
  assert(oss.str() == "test");
  
  oss << val2;
  assert(oss.str() == "testA");
 
  oss << val3;
  assert(oss.str() == "testAB");
  
  return 0;
}
