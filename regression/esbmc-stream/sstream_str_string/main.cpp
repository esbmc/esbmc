// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
using namespace std;

int main () {

  stringstream oss1, oss2, oss3, oss4;
  string mystr;
  mystr = "Sample string";

  oss1 << mystr;
  assert(oss1.str() == "Sample string");
  
  return 0;
}
