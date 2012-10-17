// find_end example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

bool myfunction (int i, int j) {
  return (i==j);
}

template<class ForwardIterator1, class ForwardIterator2>
ForwardIterator1 find_end(ForwardIterator1 first1, ForwardIterator1 last1,
		ForwardIterator2 first2, ForwardIterator2 last2) {
	if (first2 == last2)
		return last1; // specified in C++11

	ForwardIterator1 ret = last1;

	while (first1 != last1) {
		ForwardIterator1 it1 = first1;
		ForwardIterator2 it2 = first2;
		while (*it1 == *it2) {
			++it1;
			++it2;
			if (it2 == last2) {
				ret = first1;
				break;
			}
			if (it1 == last1)
				return ret;
		}
		++first1;
	}
	return ret;
}

template<class ForwardIterator1, class ForwardIterator2>
ForwardIterator1 find_end(ForwardIterator1 first1, ForwardIterator1 last1,
		ForwardIterator2 *first2, ForwardIterator2 *last2) {
	if (first2 == last2)
		return last1; // specified in C++11

	ForwardIterator1 ret = last1;

	while (first1 != last1) {
		ForwardIterator1 it1 = first1;
		ForwardIterator2 *it2 = first2;
		while (*it1 == *it2) {
			++it1;
			++it2;
			if (it2 == last2) {
				ret = first1;
				break;
			}
			if (it1 == last1)
				return ret;
		}
		++first1;
	}
	return ret;
}

template<class FwdIt1, class FwdIt2, class Pr>
FwdIt1 find_end(FwdIt1 first1, FwdIt1 last1, FwdIt2 first2, FwdIt2 last2,
		Pr pred) {
	if (first2 == last2)
		return last1; // specified in C++11

	FwdIt1 ret = last1;

	while (first1 != last1) {
		FwdIt1 it1 = first1;
		FwdIt2 it2 = first2;
		while (pred(*it1, *it2)) {
			++it1;
			++it2;
			if (it2 == last2) {
				ret = first1;
				break;
			}
			if (it1 == last1)
				return ret;
		}
		++first1;
	}
	return ret;
}

template<class FwdIt1, class FwdIt2, class Pr>
FwdIt1 find_end(FwdIt1 first1, FwdIt1 last1, FwdIt2 *first2, FwdIt2 *last2,
		Pr pred) {
	if (first2 == last2)
		return last1; // specified in C++11

	FwdIt1 ret = last1;

	while (first1 != last1) {
		FwdIt1 it1 = first1;
		FwdIt2 *it2 = first2;
		while (pred(*it1, *it2)) {
			++it1;
			++it2;
			if (it2 == last2) {
				ret = first1;
				break;
			}
			if (it1 == last1)
				return ret;
		}
		++first1;
	}
	return ret;
}

int main () {
  int myints[] = {1,2,3,4,5,1,2,3,4,5};
  vector<int> myvector (myints,myints+10);
  vector<int>::iterator it;

  int match1[] = {1,2,3};

  // using default comparison:
  it = find_end (myvector.begin(), myvector.end(), match1, match1+3);
  cout << "first *it: " << *it << endl;
  assert(*it==1);
  int match2[] = {4,5,1};

  // using predicate comparison:
  it = find_end (myvector.begin(), myvector.end(), match2, match2+3, myfunction);
  cout << "second *it: " << *it << endl;
  assert(*it!=4);
  return 0;
}
