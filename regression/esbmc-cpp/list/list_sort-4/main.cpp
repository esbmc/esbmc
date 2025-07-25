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
  mylist.push_back ("2");
  mylist.push_back ("1");

  mylist.sort();
  it = mylist.begin();
  assert(*it == "1"); 
  it++;
  assert(*it == "2");
  
  mylist.sort(compare_nocase);
  it = mylist.begin();
  assert(*it == "1"); it++;
  assert(*it == "2"); it++;

  return 0;
}
