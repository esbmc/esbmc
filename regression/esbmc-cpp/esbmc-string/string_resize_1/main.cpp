// resizing string
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  size_t sz;
  string str ("I like to code in C");
  cout << str << endl;

  sz=str.size();
  assert(int(sz) == 19);
  
  
  return 0;
}
