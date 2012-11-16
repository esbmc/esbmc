//test fails
#include <string>
#include <cassert>
using namespace std;

int main(){
  string str1, str2;
  str2 = string("Test");
  str1 = string(str2, 2);
  assert(str1 <= str2);
}
