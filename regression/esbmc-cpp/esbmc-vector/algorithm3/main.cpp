// find_if example

#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

bool IsOdd (int i) {
  return ((i%2)==1);
}

template<class InputIterator, class Predicate>
InputIterator find_if(InputIterator first, InputIterator last, Predicate pred) {
	for (; first != last; first++)
		if (pred(*first))
			break;
	return first;
}

int main () {
  vector<int> myvector;
  vector<int>::iterator it;

  myvector.push_back(10);
  myvector.push_back(25);
  myvector.push_back(40);
  myvector.push_back(55);

  it = find_if (myvector.begin(), myvector.end(), IsOdd);
  assert(*it  != 25);
  cout << "The first odd value is " << *it << endl;

  return 0;
}
