#include <vector>
#include <iostream>
using namespace std;
int main()
{
    std::vector<char> letters {'o', 'm', 'g', 'w', 't', 'f'};
 
    if (!letters.empty()) {
        cout << "The first character is: " << letters.front() << endl;
    }  
}
