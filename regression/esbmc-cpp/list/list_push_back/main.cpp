// list::push_back
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist;
  int myint;

  cout << "Please enter some integers (enter 0 to end):\n";

  do {
    cin >> myint;
    mylist.push_back (myint);
  } while (myint);

  cout << "mylist stores " << (int) mylist.size() << " numbers.\n";

  return 0;
}
