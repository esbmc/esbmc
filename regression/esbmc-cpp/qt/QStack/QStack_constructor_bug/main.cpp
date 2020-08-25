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
  first.push(1);
  assert(first.top() == 1);
  first.pop();
  first.push(2);
  assert(first.top() != 2);

  return 0;
}
