// list::rbegin/rend
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist;
  for (int i=1; i<=5; i++) mylist.push_back(i);

  cout << "mylist contains:";
  list<int>::reverse_iterator rit;
  
  rit = mylist.rbegin();
  assert(*rit != 5);
  
  for ( rit=mylist.rbegin() ; rit != mylist.rend(); ++rit )
    cout << " " << *rit;

  cout << endl;

  return 0;
}
