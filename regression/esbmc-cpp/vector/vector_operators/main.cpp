#include <vector>
#include <cassert>
using namespace std;

int main(){
	vector<int> v1(5);
	vector<int> v2(3);
	v2.push_back(0);
	v2.push_back(0);
	assert(v1 == v2);
	v1.push_back(2);
	assert(v1 != v2);
	assert(v1 >= v2);


	return 0;
}
