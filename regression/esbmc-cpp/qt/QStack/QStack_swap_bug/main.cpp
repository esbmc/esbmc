// constructing QStacks
#include <iostream>
#include <vector>
#include <deque>
#include <QStack>
#include <cassert>
using namespace std;

int main ()
{

  QStack<int> first;
  QStack<int> second;    
  first.push(1);
  assert(first.top() == 1);
  first.pop();
  first.push(2);
  assert(first.top() == 2);

  first.swap(second);

  assert(first.size() == 0);
  assert(second.size() != 1);
  return 0;
}
