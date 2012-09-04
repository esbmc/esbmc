// search algorithm example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

bool mypredicate (int i, int j) {
  return (i==j);
}

template<class FwdIt1, class FwdIt2>
FwdIt1 search(FwdIt1 first1, FwdIt1 last1, FwdIt2 first2, FwdIt2 last2) {
	if (first2 == last2)
		return first1; // specified in C++11

	while (first1 != last1) {
		FwdIt1 it1 = first1;
		FwdIt2 it2 = first2;
		while (*it1 == *it2) {
			++it1;
			++it2;
			if (it2 == last2)
				return first1;
			if (it1 == last1)
				return last1;
		}
		++first1;
	}
	return last1;
}

template<class FwdIt1, class FwdIt2>
FwdIt1 search(FwdIt1 first1, FwdIt1 last1, FwdIt2 *first2, FwdIt2 *last2) {
	if (first2 == last2)
		return first1; // specified in C++11

	while (first1 != last1) {
		FwdIt1 it1 = first1;
		FwdIt2 *it2 = first2;
		while (*it1 == *it2) {
			++it1;
			++it2;
			if (it2 == last2)
				return first1;
			if (it1 == last1)
				return last1;
		}
		++first1;
	}
	return last1;
}

template<class FwdIt1, class FwdIt2, class Pr>
FwdIt1 search(FwdIt1 first1, FwdIt1 last1, FwdIt2 first2, FwdIt2 last2,
		Pr pred) {
	if (first2 == last2)
		return first1; // specified in C++11

	while (first1 != last1) {
		FwdIt1 it1 = first1;
		FwdIt2 it2 = first2;
		while (pred(*it1, *it2)) {
			++it1;
			++it2;
			if (it2 == last2)
				return first1;
			if (it1 == last1)
				return last1;
		}
		++first1;
	}
	return last1;
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
  cout << "search *it: " << *it << endl;

  return 0;
}
