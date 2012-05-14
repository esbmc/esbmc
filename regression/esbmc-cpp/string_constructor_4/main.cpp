#include <string>
#include <cassert>

using namespace std;

int main(){
	string S = string("Testing", 3);
	string T = string();
	T = "TestingTestingTesting";
	assert(T != S);
}
