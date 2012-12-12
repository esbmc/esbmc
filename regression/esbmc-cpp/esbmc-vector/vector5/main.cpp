#include <cassert>
#include <vector>
using namespace std;

int main() {
    vector<int> vectorOne(10,5);
    vector<int> vectorTwo(vectorOne);
	 assert(vectorTwo.front() == 10);
    return 0;
}

