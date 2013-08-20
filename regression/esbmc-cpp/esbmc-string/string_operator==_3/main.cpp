#include <string>
#include <cassert>
using namespace std;

int main(){
  string str2 = string("Test");
  string str1 = string(str2);
  assert(str1 == str2);
}
