#include <string>
#include <cassert>
using namespace std;

int main(){
  string str1, str2, str3, str4;
  str1 = string("AAAA");
  str2 = string(str1, 3);
  str3 = string("AA", 6);
  str4 = string('A',12);
  assert( (str2 <= str1)&&(str1 < str3)&&(str1 <= str4) );
}
