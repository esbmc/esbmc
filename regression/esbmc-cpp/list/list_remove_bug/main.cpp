// remove from list
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  int myints[]= {17,89,7,14};
  list<int> mylist (myints,myints+4);
  list<int>::iterator it;
  
  mylist.remove(89);
  assert(mylist.size() != 3);
  it = mylist.begin(); it++;
  assert(*it != 7);

  cout << "mylist contains:";
  for (list<int>::iterator it=mylist.begin(); it!=mylist.end(); ++it)
    cout << " " << *it;
  cout << endl;

  return 0;
}
