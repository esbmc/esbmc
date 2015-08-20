// list::sort
#include <iostream>
#include <list>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  list<string> mylist;
  list<string>::iterator it;
  mylist.push_back ("one");
  mylist.push_back ("two");
  mylist.push_back ("Three");

  mylist.sort();
  it = mylist.begin();
  assert(*it != "one"); it++;
  assert(*it != "two"); it++;
  assert(*it != "Three");
  
  cout << "mylist contains:";
  for (it=mylist.begin(); it!=mylist.end(); ++it)
    cout << " " << *it;
  cout << endl;

  return 0;
}
