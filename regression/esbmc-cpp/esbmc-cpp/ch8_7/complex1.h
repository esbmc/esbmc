// Fig. 8.15: complex1.h
// Complex class definition.
#ifndef COMPLEX1_H
#define COMPLEX1_H

class Complex {

public:
   Complex( double = 0.0, double = 0.0 );       // constructor
   Complex operator+( const Complex & ) const;  // addition
   Complex operator-( const Complex & ) const;  // subtraction
   void print() const;                          // output

private:
   double real;       // real part
   double imaginary;  // imaginary part

}; // end class Complex

#endif

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
