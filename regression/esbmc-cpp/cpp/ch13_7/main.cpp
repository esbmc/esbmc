// Fig. 13.7: fig13_07.cpp
// Demonstrating auto_ptr.
#include <iostream>

using std::cout;
using std::endl;

#include <memory>

using std::auto_ptr; // auto_ptr class definition

class Integer {

public:

   // Integer constructor
   Integer( int i = 0 )
      : value( i )
   {
      cout << "Constructor for Integer " << value << endl;

   } // end Integer constructor

   // Integer destructor
   ~Integer()
   {
      cout << "Destructor for Integer " << value << endl;

   } // end Integer destructor

   // functions to set Integer
   void setInteger( int i )
   {
      value = i;

   } // end function setInteger

   // function to return Integer
   int getInteger() const
   {
      return value;

   } // end function getInteger

private:
   int value;

};  // end class Integer

// use auto_ptr to manipulate Integer object
int main()
{
   cout << "Creating an auto_ptr object that points to an "
        << "Integer\n";

   // "aim" auto_ptr at Integer object
   auto_ptr< Integer > ptrToInteger( new Integer( 7 ) );

   cout << "\nUsing the auto_ptr to manipulate the Integer\n";

   // use auto_ptr to set Integer value
   ptrToInteger->setInteger( 99 );

   // use auto_ptr to get Integer value
   cout << "Integer after setInteger: " 
        << ( *ptrToInteger ).getInteger()
        << "\n\nTerminating program" << endl;

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