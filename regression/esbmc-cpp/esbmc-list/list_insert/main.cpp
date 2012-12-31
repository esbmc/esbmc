// inserting into a list
#include <iostream>
#include <list>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist;
  list<int>::iterator it;

  // set some initial values:
  for (int i=1; i<=5; i++) mylist.push_back(i); // 1 2 3 4 5

  it = mylist.begin();
  ++it;       // it points now to number 2           ^

  mylist.insert (it,10);                        // 1 10 2 3 4 5
  assert(*it == 2);
  assert(mylist.size() == 6);

  // "it" still points to number 2                      ^
  mylist.insert (it,2,20);                      // 1 10 20 20 2 3 4 5

  --it;       // it points now to the second 20            ^
  assert(*it == 20);
  assert(mylist.size() == 8);

  vector<int> myvector (2,30);
  mylist.insert (it,myvector.begin(),myvector.end());
                                                // 1 10 20 30 30 20 2 3 4 5
                                                //               ^
  it--;
  assert(mylist.size() == 10);
  assert(*it == 30);
  cout << "mylist contains:";
  for (it=mylist.begin(); it!=mylist.end(); it++)
    cout << " " << *it;
  cout << endl;

  return 0;
}
