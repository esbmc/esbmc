#include <map>
#include <string>
#include <tuple>

using namespace std;

int main() {
    map<tuple<string>, int> x;
    tuple<string> tmp;

    x[tmp];

    return 0;
}

