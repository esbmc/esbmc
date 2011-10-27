// Fig 22.6: array.h
// Simple class Array (for integers).
#ifndef ARRAY_H
#define ARRAY_H

#include <iostream>

using std::ostream;

// class Array definition
class Array {
   friend ostream &operator<<( ostream &, const Array & );
public:
   Array( int = 10 );  // default/conversion constructor
   ~Array();           // destructor
private:
   int size; // size of the array
   int *ptr; // pointer to first element of array

};  // end class Array

#endif  // ARRAY_H

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
