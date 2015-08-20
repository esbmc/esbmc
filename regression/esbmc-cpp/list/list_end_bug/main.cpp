// list::begin/end
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  int myints[] = {75,23,65,42,13};
  list<int> mylist (myints,myints+5);

  list<int>::iterator it;
  it = mylist.end();
  it--;
  assert(*it != 13);
  
  cout << endl;

  return 0;
}
