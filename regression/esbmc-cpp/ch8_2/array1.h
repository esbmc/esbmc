// Fig. 8.4: array1.h
// Array class for storing arrays of integers.
#ifndef ARRAY1_H
#define ARRAY1_H

#include <iostream>

using std::ostream;
using std::istream;

class Array {
   friend ostream &operator<<( ostream &, const Array & );
   friend istream &operator>>( istream &, Array & );

public:
   Array( int = 10 );       // default constructor
   Array( const Array & );  // copy constructor
   ~Array();                // destructor
   int getSize() const;     // return size

   // assignment operator
   const Array &operator=( const Array & ); 
   
   // equality operator
   bool operator==( const Array & ) const;  

   // inequality operator; returns opposite of == operator
   bool operator!=( const Array &right ) const  
   { 
      return ! ( *this == right ); // invokes Array::operator==
   
   } // end function operator!=
   
   // subscript operator for non-const objects returns lvalue
   int &operator[]( int );              

   // subscript operator for const objects returns rvalue
   const int &operator[]( int ) const;  
   
private:
   int size; // array size
   int *ptr; // pointer to first element of array

}; // end class Array

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