#include <string>
#include <cassert>
#include <iostream>
using namespace std;

int main(){
  string str2 = string("Test");
  string str1 = str2 + str2;
  assert(str1 != str2);
}
