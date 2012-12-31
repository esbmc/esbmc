// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
using namespace std;

int main () {

  stringstream oss;
  long val1 = 1334212;
  unsigned long val2 = 23132123;
  
  oss << val1;
  
  assert(oss.str() == "1334212");
  
  oss << val2;
  assert(oss.str() == "133421223132123");
  
  return 0;
}
