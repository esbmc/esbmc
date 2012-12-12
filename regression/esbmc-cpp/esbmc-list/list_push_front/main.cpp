// list::push_front
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist (2,100);         // two ints with a value of 100
  assert(mylist.front() == 100);
  mylist.push_front (200);
  assert(mylist.front() == 200);
  mylist.push_front (300);
  assert(mylist.front() == 300);

  cout << "mylist contains:";
  for (list<int>::iterator it=mylist.begin(); it!=mylist.end(); ++it)
    cout << " " << *it;

  cout << endl;
  return 0;
}
