// Fig. 22.14: base1.h
// Definition of class Base1
#ifndef BASE1_H
#define BASE1_H

// class Base1 definition
class Base1 {
public:
   Base1( int parameterValue ) { value = parameterValue; }
   int getData() const { return value; }

protected:      // accessible to derived classes
   int value;   // inherited by derived class

};  // end class Base1

#endif  // BASE1_H

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
