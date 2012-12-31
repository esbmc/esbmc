// list::pop_front
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist;
  mylist.push_back (100);
  mylist.push_back (200);
  mylist.push_back (300);
  assert(mylist.front() == 100);
  
  int n = 100;
  
  cout << "Popping out the elements in mylist:";
  while (!mylist.empty())
  {
    assert(mylist.front() == n);
    cout << " " << mylist.front();
    mylist.pop_front();
    n +=100;
  }
  assert(mylist.empty());
  cout << "\nFinal size of mylist is " << int(mylist.size()) << endl;

  return 0;
}
