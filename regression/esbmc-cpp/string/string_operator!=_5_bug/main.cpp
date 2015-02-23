//TEST FAILS
#include <string>
#include <cassert>
using namespace std;

int main(){
  string str1 = string();
  str1 = 'T';
  assert(str1 != "T");
}
