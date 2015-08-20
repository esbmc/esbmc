#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  int myints[] = {75,23,65,42,13};
  set<int> myset (myints,myints+5);

  set<int>::iterator it = myset.end();
  it--;
  assert(*it == 75);


  return 0;
}
