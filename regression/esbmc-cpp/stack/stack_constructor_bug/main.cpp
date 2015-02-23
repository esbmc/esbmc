// constructing stacks
#include <iostream>
#include <vector>
#include <deque>
#include <stack>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> mydeque (3,100);     // deque with 3 elements
  vector<int> myvector (2,200);   // vector with 2 elements

  stack<int> first;               // empty stack
  stack<int> second (mydeque);    // stack initialized to copy of deque

  stack<int,vector<int> > third;  // empty stack using vector
  stack<int,vector<int> > fourth (myvector);
  
  assert(first.size() == 0);
  assert(second.size() != 3);
  assert(third.size() == 0);
  assert(fourth.size() != 2);

  cout << "size of first: " << (int) first.size() << endl;
  cout << "size of second: " << (int) second.size() << endl;
  cout << "size of third: " << (int) third.size() << endl;
  cout << "size of fourth: " << (int) fourth.size() << endl;

  return 0;
}
