// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
using namespace std;

int main () {

  stringstream oss;
  bool val = true;
  
  oss << val;
  assert(oss.str() == "1");
  
  oss << boolalpha << val;
  assert(oss.str() == "1true");
  
  return 0;
}
