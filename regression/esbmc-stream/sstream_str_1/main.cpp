// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
using namespace std;

int main () {

  stringstream oss1, oss2, oss3, oss4;
  string mystr;

  oss1 << "Sample string";
  assert(oss1.str() == "Sample string");
  
  oss2 << 1532;
  assert(oss2.str() == "1532");
  
  oss3 << 14.4;
  assert(oss3.str() == "14.4");
  
  oss4 << 'D';
  assert(oss4.str() == "D");


  return 0;
}
