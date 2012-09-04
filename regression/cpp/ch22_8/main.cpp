// Fig. 22.3: fig22_03.cpp
// Demonstrating namespaces.
#include <iostream>

using namespace std;  // use std namespace

int integer1 = 98;  // global variable

// create namespace Example
namespace Example {

   // declare two constants and one variable
   const double PI = 3.14159;
   const double E = 2.71828; 
   int integer1 = 8;

   void printValues();  // prototype

   // nested namespace
   namespace Inner { 

      // define enumeration
      enum Years { FISCAL1 = 1990, FISCAL2, FISCAL3 };

   }  // end Inner

}  // end Example

// create unnamed namespace
namespace {           
   double doubleInUnnamed = 88.22;  // declare variable

}  // end unnamed namespace

int main()
{
   // output value doubleInUnnamed of unnamed namespace
   cout << "doubleInUnnamed = " << doubleInUnnamed;

   // output global variable
   cout << "\n(global) integer1 = " << integer1;

   // output values of Example namespace
   cout << "\nPI = " << Example::PI << "\nE = "
        << Example::E << "\ninteger1 = " 
        << Example::integer1 << "\nFISCAL3 = "
        << Example::Inner::FISCAL3 << endl;

   Example::printValues();  // invoke printValues function

   return 0;

}  // end main

// display variable and constant values
void Example::printValues() 
{
   cout << "\nIn printValues:\ninteger1 = "
        << integer1 << "\nPI = " << PI << "\nE = "
        << E << "\ndoubleInUnnamed = " << doubleInUnnamed 
        << "\n(global) integer1 = " << ::integer1 
        << "\nFISCAL3 = " << Inner::FISCAL3 << endl;

}  // end printValues

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
