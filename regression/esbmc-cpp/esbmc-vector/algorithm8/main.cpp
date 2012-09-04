// count_if example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

bool IsOdd (int i) { return ((i%2)==1); }

template<class InputIterator, class T>
ptrdiff_t count_if(InputIterator first, InputIterator last, const T& pred) {
	ptrdiff_t ret = 0;
	while (first != last)
		if (pred(*first++))
			++ret;
	return ret;
}

int main () {
  int mycount;

  vector<int> myvector;
  for (int i=1; i<10; i++) myvector.push_back(i); // myvector: 1 2 3 4 5 6 7 8 9

  mycount = (int) count_if (myvector.begin(), myvector.end(), IsOdd);
  assert(mycount == 5);
  cout << "myvector contains " << mycount  << " odd values.\n";

  return 0;
}
