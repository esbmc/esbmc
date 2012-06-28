#include <cassert>
#include <iostream>
using namespace std; 
class exforsys
{
public:
        exforsys(void) { x=0; }
        void f(int n1)
        {
                x= n1*5;
        } 
 
        void output(void) { cout << "n" << "x=" << x << endl; } 
 		  int getX() { return x; }
private:
        int x;
}; 
 
class sample: virtual public exforsys
{
public:
        sample(void) { s1=0; } 
 
        void f1(int n1)
        {
                s1=n1*10;
        } 
 
        void output(void)
        {
                exforsys::output();
                cout << "n" << "s1=" << s1 << endl;
        } 
 
private:
        int s1;
}; 
 
int main(void)
{
        sample s;
        s.f(10);
		  assert(s.getX() == 30);
        s.f1(20);
        s.output();
}
