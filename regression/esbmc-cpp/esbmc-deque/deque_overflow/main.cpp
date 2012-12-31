// deque::operator[]
#include <iostream>
#include <deque>
using namespace std;

int main ()
{
  deque<int> mydeque (10);   // 10 zero-initialized elements
  unsigned int i;

  deque<int>::size_type sz = mydeque.size();
  sz += 3;
  // assign some values:
  for (i=0; i<sz; i++) mydeque[i]=i;

  // reverse order of elements using operator[]:
  for (i=0; i<sz/2; i++)
  {
    int temp;
    temp = mydeque[sz-1-i];
    mydeque[sz-1-i]=mydeque[i];
    mydeque[i]=temp;
  }

  cout << "mydeque contains:";
  for (i=0; i<sz; i++)
    cout << " " << mydeque[i];
  mydeque[25] = 2;

  cout << endl;

  return 0;
}
