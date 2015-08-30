// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
using namespace std;

int main () {

  stringstream oss;
  int val1 = -1532;
  unsigned int val2 = 433;
  
  oss << val1;
  assert(oss.str() == "-1532");
  
  
  oss << val2;
  assert(oss.str() == "-1532433");
  
  return 0;
}
