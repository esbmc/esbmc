// list::empty
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist;
  int sum (0);

  for (int i=1;i<=10;i++) mylist.push_back(i);

  assert(!mylist.empty());
  while (!mylist.empty())
  {
     sum += mylist.front();
     mylist.pop_front();
  }
  assert(!mylist.empty()||(mylist.size() != 0));
  cout << "total: " << sum << endl;
  
  return 0;
}
