// stringstream::str
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstring>
using namespace std;

int main () {

  stringstream oss;
  char* val1 = new char[10];
  strcpy(val1, "testasd");

  oss << val1;
  assert(oss.str() == "testasd");


  return 0;
}
