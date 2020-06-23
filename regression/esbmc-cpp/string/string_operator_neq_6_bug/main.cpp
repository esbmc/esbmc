//TEST FAILS
#include <string>
#include <cassert>
using namespace std;

int main(){
  string str1 = string();
  string aux = str1;
  string str2 = string("Test1");
  str1 = str2;
  aux = str2;
  assert(str1 != aux);
}
