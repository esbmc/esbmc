// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
using namespace std;

int main () {

  stringstream oss;
  double val1 = 23.13;
  long double val2 = 45.543;
  
  oss << val1;
  assert(oss.str() == "23.13");
  
  oss << val2;
  assert(oss.str() == "23.1345.543");
  
  
  return 0;
}
