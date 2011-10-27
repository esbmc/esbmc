// Fig. 8.18: hugeint1.h 
// HugeInt class definition.
#ifndef HUGEINT1_H
#define HUGEINT1_H

#include <iostream>

using std::ostream;

class HugeInt {
   friend ostream &operator<<( ostream &, const HugeInt & );

public:
   HugeInt( long = 0 );      // conversion/default constructor
   HugeInt( const char * );  // conversion constructor

   // addition operator; HugeInt + HugeInt
   HugeInt operator+( const HugeInt & );

   // addition operator; HugeInt + int
   HugeInt operator+( int );            

   // addition operator; 
   // HugeInt + string that represents large integer value
   HugeInt operator+( const char * );    

private:
   short integer[ 30 ];

}; // end class HugeInt

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