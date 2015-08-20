// search algorithm example
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

bool mypredicate (int i, int j) {
  return (i==j);
}

int main () {
  vector<int> myvector;
  vector<int>::iterator it;

  // set some values:        myvector: 10 20 30 40 50 60 70 80 90
  for (int i=1; i<10; i++) myvector.push_back(i*10);


  // using default comparison:
  int match1[] = {40,50,60,70};
  it = search (myvector.begin(), myvector.end(), match1, match1+4);
  assert(*it == 40);

  return 0;
}
