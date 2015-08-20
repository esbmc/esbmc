//TEST FAILS
#include <string>
#include <cassert>
using namespace std;

int main(){
  string aux = string("Test");
  string str1, str2;
  str1 = 'D';
  str2 = string(str1);
  assert((aux <= str2)&&(aux <= str1));
}
