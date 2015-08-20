// Fig 22.7: array.cpp
// Member function definitions for class Array.
#include <iostream>

using std::cout;
using std::ostream;

#include <new>

#include "array.h"

// default constructor for class Array (default size 10)
Array::Array( int arraySize )
{
   size = ( arraySize < 0 ? 10 : arraySize ); 
   cout << "Array constructor called for " 
        << size << " elements\n";

   // create space for array
   ptr = new int[ size ];

   // initialize array elements to zeroes
   for ( int i = 0; i < size; i++ )
      ptr[ i ] = 0;          

}  // end constructor

// destructor for class Array
Array::~Array() { delete [] ptr; }

// overloaded stream insertion operator for class Array 
ostream &operator<<( ostream &output, const Array &arrayRef )
{
   for ( int i = 0; i < arrayRef.size; i++ )
      output << arrayRef.ptr[ i ] << ' ' ;

   return output;   // enables cout << x << y;

}  // end operator<<

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
