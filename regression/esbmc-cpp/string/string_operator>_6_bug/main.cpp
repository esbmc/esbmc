//TEST FAILS
#include <string>
#include <cassert>
using namespace std;

int main(){
  string aux;
  aux = 'D';
  string str1, str2;
  str1 = string("Test");
  str2 = string(str1, 2);
  assert((aux > str2)&&(aux > str1));
}
