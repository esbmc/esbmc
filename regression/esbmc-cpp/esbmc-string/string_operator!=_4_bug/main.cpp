//TEST FAILS
#include <string>
#include <cassert>
using namespace std;

int main(){
  string str1 = string();
  str1 = "Test";
  assert(str1 != "Test");
}
