// Fig. 22.20: fig22_20.cpp
// Attempting to polymorphically call a function that is
// multiply inherited from two base classes.
#include <iostream>

using std::cout;
using std::endl;

// class Base definition
class Base {
public:   
   virtual void print() const = 0;  // pure virtual

};  // end class Base

// class DerivedOne definition
class DerivedOne : public Base {
public:

   // override print function
   void print() const { cout << "DerivedOne\n"; }
   
};  // end class DerivedOne

// class DerivedTwo definition
class DerivedTwo : public Base {
public:

   // override print function
   void print() const { cout << "DerivedTwo\n"; }

};  // end class DerivedTwo

// class Multiple definition
class Multiple : public DerivedOne, public DerivedTwo {
public:

   // qualify which version of function print
   void print() const { DerivedTwo::print(); }

};  // end class Multiple

int main()
{
   Multiple both;   // instantiate Multiple object
   DerivedOne one;  // instantiate DerivedOne object
   DerivedTwo two;  // instantiate DerivedTwo object

   // create array of base-class pointers
   Base *array[ 3 ];

   array[ 0 ] = &both;   // ERROR--ambiguous
   array[ 1 ] = &one;
   array[ 2 ] = &two;

   // polymorphically invoke print
   for ( int i = 0; i < 3; i++ )
      array[ i ] -> print();

   return 0;

}  // end main

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/
