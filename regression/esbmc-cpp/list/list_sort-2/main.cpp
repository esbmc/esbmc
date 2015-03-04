// list::sort
#include <iostream>
#include <list>
#include <string>
#include <cctype>
#include <cassert>
using namespace std;

// comparison, not case sensitive.
bool compare_nocase (string first, string second)
{
  unsigned int i=0;
  while ( (i<first.length()) && (i<second.length()) )
  {
    if (tolower(first[i])<tolower(second[i])) return true;
    else if (tolower(first[i])>tolower(second[i])) return false;
    ++i;
  }
  if (first.length()<second.length()) return true;
  else return false;
}

int main ()
{
  list<string> mylist;
  list<string>::iterator it;
  mylist.push_back ("one");
  mylist.push_back ("two");
  mylist.push_back ("Three");

  mylist.sort();
  it = mylist.begin();
  assert(*it == "Three"); it++;
  assert(*it == "one"); it++;
  assert(*it == "two");
  
  cout << "mylist contains:";
  for (it=mylist.begin(); it!=mylist.end(); ++it)
    cout << " " << *it;
  cout << endl;

  mylist.sort(compare_nocase);
  it = mylist.begin();
  assert(*it == "one"); it++;
  assert(*it == "Three"); it++;
  assert(*it == "two");

  cout << "mylist contains:";
  for (it=mylist.begin(); it!=mylist.end(); ++it)
    cout << " " << *it;
  cout << endl;

  return 0;
}
