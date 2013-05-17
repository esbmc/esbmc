#include <string>
#include <cassert>
using namespace std;

int main(){
  string str1 = string("Test", 2);
  string str2 = string("TestTest");
  string str3 = string();

  string strN = " ";
  str3 = string(str1 + str2 + strN);
  assert(str3 != "TestTestTestTest");
}
