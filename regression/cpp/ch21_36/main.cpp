// stack::size
#include <iostream>
#include <stack>
#include <cassert>
using namespace std;

int main ()
{
  stack<int> myints;
  cout << "0. size: " << (int) myints.size() << endl;
  assert(myints.size()==0);
  for (int i=0; i<5; i++) myints.push(i);
  cout << "1. size: " << (int) myints.size() << endl;
  assert(myints.size()==5);
  myints.pop();
  cout << "2. size: " << (int) myints.size() << endl;
  assert(myints.size()==4);
  return 0;
}

