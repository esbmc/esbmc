#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  set<int> myset;
  set<int>::iterator it,itlow,itup;

  for (int i=1; i<10; i++) myset.insert(i*10); // 10 20 30 40 50 60 70 80 90
  assert(myset.size() == 9);
  itlow=myset.lower_bound (30);                //       ^
  assert(*itlow == 30);
  itup=myset.upper_bound (60);                 //                   ^
  assert(*itup == 70);
  myset.erase(itlow,itup);                     // 10 20 70 80 90
  assert(myset.size() == 5);
  cout << "myset contains:";
  for (it=myset.begin(); it!=myset.end(); it++)
    cout << " " << *it;
  cout << endl;

  return 0;
}
