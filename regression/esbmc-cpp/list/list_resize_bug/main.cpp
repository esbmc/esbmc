//TEST FAILS
// resizing list
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist;

  unsigned int i;

  // set some initial content:
  for (i=1;i<10;i++) mylist.push_back(i);
  assert(mylist.size() == 9);
  mylist.resize(5);
  assert(mylist.size() == 5);  
  mylist.resize(8,100);
  assert(mylist.size() != 8);  
  assert(mylist.back() != 100);
  mylist.resize(12);
  assert(mylist.size() == 12);
  assert(mylist.back() != 0);

  cout << "mylist contains:";
  for (list<int>::iterator it=mylist.begin();it!=mylist.end();++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
