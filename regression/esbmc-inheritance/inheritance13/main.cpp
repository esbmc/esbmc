#include<iostream>
using namespace std;
//#include<conio.h>
 
class student
{
    protected:
       int rno,m1,m2;
    public:
                void get()
              {
                            cout<<"Enter the Roll no :";
                            cin>>rno;
                            cout<<"Enter the two marks   :";
                            cin>>m1>>m2;
              }
};
class sports
{
    protected:
       int sm;                   // sm = Sports mark
    public:
                void getsm()
              {
                 cout<<"\nEnter the sports mark :";
                 cin>>sm;
 
              }
};
class statement:public student,public sports
{
    int tot,avg;
    public:
    void display()
              {
                 tot=(m1+m2+sm);
                 avg=tot/3;
                 cout<<"\n\n\tRoll No    : "<<rno<<"\n\tTotal      : "<<tot;
               cout<<"\n\tAverage    : "<<avg;
              }
};
int main()
{
   statement obj;
   obj.get();
   obj.getsm();
   obj.display();
return 0;
}
