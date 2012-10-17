#include <string>
#include <cassert>

using namespace std;

int main(){
	string S = string();
	S = "Testing";
	string T = string(S);
	assert(T == S);
}
