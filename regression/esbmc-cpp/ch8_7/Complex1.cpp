// Fig. 8.16: complex1.cpp
// Complex class member function definitions.
#include <iostream>

using std::cout;

#include "complex1.h"  // Complex class definition

// Constructor
Complex::Complex( double realPart, double imaginaryPart ) 
   : real( realPart ), 
     imaginary( imaginaryPart ) 
{ 
   // empty body

} // end Complex constructor

// addition operator
Complex Complex::operator+( const Complex &operand2 ) const
{
   return Complex( real + operand2.real, 
      imaginary + operand2.imaginary );

} // end function operator+

// subtraction operator
Complex Complex::operator-( const Complex &operand2 ) const
{
   return Complex( real - operand2.real, 
      imaginary - operand2.imaginary );

} // end function operator-

// display a Complex object in the form: (a, b)
void Complex::print() const
{ 
   cout << '(' << real << ", " << imaginary << ')'; 

} // end function print

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