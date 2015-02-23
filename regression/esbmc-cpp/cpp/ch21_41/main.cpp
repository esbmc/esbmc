// list::front
#include <iostream>
#include <list>
#include <cassert>

using namespace std;

int main ()
{
  list<int> mylist;

  mylist.push_back(77);
  mylist.push_back(22);

  // now front equals 77, and back 22


  cout << "mylist.front() is now " << mylist.front() << endl;
  cout << "mylist.back() is now " << mylist.back() << endl;

  assert(mylist.front()==77);
  assert(mylist.back()==22);

  return 0;
}


