#include <algorithm>
#include <iostream>

using namespace std;

class Cat                      // begin declaration of the class
{
  public:                      // begin public section
    Cat(int initialAge);       // constructor
    Cat(const Cat& copy_from); //copy constructor
    Cat& operator=(const Cat& copy_from); //copy assignment
    ~Cat();                    // destructor

    int GetAge() const;        // accessor function
    void SetAge(int age);      // accessor function
    void Meow();
 private:                      // begin private section
    int itsAge;                // member variable
    char * string;
};

// constructor of Cat,
Cat::Cat(int initialAge)
{
  itsAge = initialAge;
  string = new char[10]();
}

//copy constructor for making a new copy of a Cat
Cat::Cat(const Cat& copy_from) {
   itsAge = copy_from.itsAge;
   string = new char[10]();
   std::copy(copy_from.string+0, copy_from.string+10, string);
}

//copy assignment for assigning a value from one Cat to another
Cat& Cat::operator=(const Cat& copy_from) {
   itsAge = copy_from.itsAge;
   std::copy(copy_from.string+0, copy_from.string+10, string);
   return *this;
}

// destructor, just an example
Cat::~Cat()
{
    delete[] string;
}

// GetAge, Public accessor function
// returns value of itsAge member
int Cat::GetAge() const
{
   return itsAge;
}

// Definition of SetAge, public
// accessor function
 void Cat::SetAge(int age)
{
   // set member variable its age to
   // value passed in by parameter age
   itsAge = age;
}

// definition of Meow method
// returns: void
// parameters: None
// action: Prints "meow" to screen
void Cat::Meow()
{
   cout << "Meow.\n";
}

// create a cat, set its age, have it
// meow, tell us its age, then meow again.
int main()
{
  int Age;
  cout<<"How old is Frisky? ";
  cin>>Age;
  Cat Frisky(Age);
  Frisky.Meow();
  cout << "Frisky is a cat who is " ;
  cout << Frisky.GetAge() << " years old.\n";
  Frisky.Meow();
  Age++;
  Frisky.SetAge(Age);
  cout << "Now Frisky is " ;
  cout << Frisky.GetAge() << " years old.\n";
  return 0;
}

