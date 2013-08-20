// reversing list
#include <iostream>
#include <cassert>
#include <list>
using namespace std;

int main ()
{
  list<int> mylist;
  list<int>::iterator it;

  for (int i=1; i<5; i++) mylist.push_back(i);

  mylist.reverse();

  it = mylist.begin();
  assert(*it != 4);

  cout << "*it: " << *it << endl;
  return 0;
}
