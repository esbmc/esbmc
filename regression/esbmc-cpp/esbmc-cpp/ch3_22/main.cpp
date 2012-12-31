// Fig. 3.25: fig03_25.cpp
// Using overloaded functions.
#include <iostream>

using std::cout;
using std::endl;

// function square for int values
int square( int x ) 
{ 
   cout << "Called square with int argument: " << x << endl;
   return x * x; 

} // end int version of function square

// function square for double values
double square( double y ) 
{ 
   cout << "Called square with double argument: " << y << endl;
   return y * y; 

} // end double version of function square

int main()
{
   int intResult = square( 7 );         // calls int version
   double doubleResult = square( 7.5 ); // calls double version

   cout << "\nThe square of integer 7 is " << intResult
        << "\nThe square of double 7.5 is " << doubleResult 
        << endl;    

   return 0;  // indicates successful termination

} // end main

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
