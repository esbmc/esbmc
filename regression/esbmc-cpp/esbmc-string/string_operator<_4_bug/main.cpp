//TEST FAILS
#include <string>
#include <cassert>
using namespace std;

int main(){
  string str1, str2;
  str1 = string("Test");
  str2 = string(str1, 2);
  assert(str2 < str1);
}
