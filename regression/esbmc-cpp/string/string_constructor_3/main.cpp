#include <string>
#include <cassert>

using namespace std;

int main(){
	string S = string("Testing");
	string T = string();
	T = "Testing again";
	assert(T != S);
}
