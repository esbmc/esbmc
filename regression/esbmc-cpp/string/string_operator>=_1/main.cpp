#include <string>
#include <cassert>
#include <iostream>
using namespace std;

int main(){
  string str1, str2;
  str1 = string("Test");
  str2 = string(str1, 2);
  assert(str1 >= str2);
}
