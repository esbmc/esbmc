#include <cassert>
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

int main() {
    vector<int> vectorOne(10);
    assert(vectorOne.size() == 10);
    assert(vectorOne.capacity() == 10);
    cout << "Size of vectorOne is: " << vectorOne.size() << " elements." << endl;
    cout << "Capacity of vectorOne is: " << vectorOne.capacity() << " elements." << endl;
    for (long index=0; index<(long)vectorOne.size(); ++index) vectorOne.at(index)=rand();
    cout << "vectorOne contains the following elements:" << endl;
    for (long index=0; index<(long)vectorOne.size(); ++index) {
//		  assert(vectorOne.at(index) == rand());
        cout << vectorOne.at(index) << " ";
    }
    cout << endl << endl;
    cout << "Using reserve to reallocate vectorOne with enough storage for 40 elements."  << endl;
    vectorOne.reserve(40);
    assert(vectorOne.size() == 10);
    assert(vectorOne.capacity() == 40);
    cout << "Size of vectorOne is: " << vectorOne.size() << " elements." << endl;
    cout << "Capacity of vectorOne is: " << vectorOne.capacity() << " elements." << endl;
    for (long index=0; index<(long)vectorOne.size(); ++index) vectorOne.at(index)=rand();
    cout << "vectorOne contains the following elements:" << endl;
    for (long index=0; index<(long)vectorOne.size(); ++index) {
        cout << vectorOne.at(index) << " ";
    }
    cout << endl << endl;
    cout << "Using resize to increase size of vector to 15 elements, with new elements set to 0." << endl;
    vectorOne.resize(15,(int)0);
    assert(vectorOne.size() == 15);
    assert(vectorOne.capacity() == 40);
    cout << "Size of vectorOne is: " << vectorOne.size() << " elements." << endl;
    cout << "Capacity of vectorOne is: " << vectorOne.capacity() << " elements." << endl;
    for (long index=0; index<(long)vectorOne.size(); ++index) vectorOne.at(index)=rand();
    cout << "vectorOne contains the following elements:" << endl;
    for (long index=0; index<(long)vectorOne.size(); ++index) {
        cout << vectorOne.at(index) << " ";
    }
    cout << endl << endl;
    cout << "Using resize to decrease size of vector to 5 elements." << endl;
    vectorOne.resize(5);
    assert(vectorOne.size() == 5);
    assert(vectorOne.capacity() == 40);
    cout << "Size of vectorOne is: " << vectorOne.size() << " elements." << endl;
    cout << "Capacity of vectorOne is: " << vectorOne.capacity() << " elements." << endl;
    for (long index=0; index<(long)vectorOne.size(); ++index) vectorOne.at(index)=rand();
    cout << "vectorOne contains the following elements:" << endl;
    for (long index=0; index<(long)vectorOne.size(); ++index) {
        cout << vectorOne.at(index) << " ";
    }
    cout << endl << endl;
    return 0;
}
