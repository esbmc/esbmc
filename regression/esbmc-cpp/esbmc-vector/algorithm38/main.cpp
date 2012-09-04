// partition algorithm example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class Ty>
void swap_esbmc(Ty& a, Ty& b) {
	Ty c(a);
	a = b;
	b = c;
}

template<class BidIt, class Pr>
BidIt partition(BidIt first, BidIt last, Pr pred) {
	while (true) {
		while (first != last && pred(*first))
			++first;
		if (first == last--)
			break;
		while (first != last && !pred(*last))
			--last;
		if (first == last)
			break;
		swap_esbmc(*first++, *last);
	}
	return first;
}

bool IsOdd (int i) { return (i%2)==1; }

int main () {
  vector<int> myvector;
  vector<int>::iterator it, bound;

  // set some values:
  for (int i=1; i<10; ++i) myvector.push_back(i); // 1 2 3 4 5 6 7 8 9

  bound = partition (myvector.begin(), myvector.end(), IsOdd);

  // print out content:
  cout << "odd members:";
  for (it=myvector.begin(); it!=bound; ++it)
    cout << " " << *it;

  assert(*bound == 6);
  cout << "\neven members:";
  for (it=bound; it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
