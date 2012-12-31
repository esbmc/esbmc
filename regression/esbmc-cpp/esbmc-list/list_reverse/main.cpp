// reversing list
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist;
  list<int>::iterator it;

  for (int i=1; i<10; i++) mylist.push_back(i);

  mylist.reverse();
  int n = 9;
  cout << "mylist contains:";
  for (it=mylist.begin(); it!=mylist.end(); ++it)
      assert(*it == n--);

  cout << endl;

  return 0;
}
