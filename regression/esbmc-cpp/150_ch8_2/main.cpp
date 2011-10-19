// Fig 8.5: array1.cpp
// Member function definitions for class Array
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include <iomanip>

using std::setw;

#include <new>       // C++ standard "new" operator

#include <cstdlib>   // exit function prototype

#include "array1.h"  // Array class definition

// default constructor for class Array (default size 10)
Array::Array( int arraySize )
{
   // validate arraySize
   size = ( arraySize > 0 ? arraySize : 10 ); 

   ptr = new int[ size ]; // create space for array

   for ( int i = 0; i < size; i++ )
      ptr[ i ] = 0;          // initialize array

} // end Array default constructor

// copy constructor for class Array;
// must receive a reference to prevent infinite recursion
Array::Array( const Array &arrayToCopy ) 
   : size( arrayToCopy.size )
{
   ptr = new int[ size ]; // create space for array

   for ( int i = 0; i < size; i++ )
      ptr[ i ] = arrayToCopy.ptr[ i ]; // copy into object

} // end Array copy constructor

// destructor for class Array
Array::~Array()
{
   delete [] ptr;            // reclaim array space

} // end destructor

// return size of array
int Array::getSize() const
{
   return size;

} // end function getSize

// overloaded assignment operator;
// const return avoids: ( a1 = a2 ) = a3
const Array &Array::operator=( const Array &right )
{
   if ( &right != this ) {  // check for self-assignment
      
      // for arrays of different sizes, deallocate original
      // left-side array, then allocate new left-side array
      if ( size != right.size ) {
         delete [] ptr;         // reclaim space
         size = right.size;     // resize this object
         ptr = new int[ size ]; // create space for array copy
         
      } // end inner if

      for ( int i = 0; i < size; i++ )
         ptr[ i ] = right.ptr[ i ];  // copy array into object

   } // end outer if

   return *this;   // enables x = y = z, for example

} // end function operator=

// determine if two arrays are equal and
// return true, otherwise return false
bool Array::operator==( const Array &right ) const
{
   if ( size != right.size )
      return false;    // arrays of different sizes

   for ( int i = 0; i < size; i++ )

      if ( ptr[ i ] != right.ptr[ i ] )
         return false; // arrays are not equal

   return true;        // arrays are equal

} // end function operator==

// overloaded subscript operator for non-const Arrays
// reference return creates an lvalue
int &Array::operator[]( int subscript )
{
   // check for subscript out of range error
   if ( subscript < 0 || subscript >= size ) {
      cout << "\nError: Subscript " << subscript 
           << " out of range" << endl;

      exit( 1 );  // terminate program; subscript out of range

   } // end if

   return ptr[ subscript ]; // reference return

} // end function operator[]

// overloaded subscript operator for const Arrays
// const reference return creates an rvalue
const int &Array::operator[]( int subscript ) const
{
   // check for subscript out of range error
   if ( subscript < 0 || subscript >= size ) {
      cout << "\nError: Subscript " << subscript 
           << " out of range" << endl;

      exit( 1 );  // terminate program; subscript out of range

   } // end if

   return ptr[ subscript ]; // const reference return

} // end function operator[]

// overloaded input operator for class Array;
// inputs values for entire array
istream &operator>>( istream &input, Array &a )
{
   for ( int i = 0; i < a.size; i++ )
      input >> a.ptr[ i ];

   return input;   // enables cin >> x >> y;

} // end function 

// overloaded output operator for class Array 
ostream &operator<<( ostream &output, const Array &a )
{
   int i;

   // output private ptr-based array 
   for ( i = 0; i < a.size; i++ ) {
      output << setw( 12 ) << a.ptr[ i ];

      if ( ( i + 1 ) % 4 == 0 ) // 4 numbers per row of output
         output << endl;

   } // end for

   if ( i % 4 != 0 )  // end last line of output
      output << endl;

   return output;   // enables cout << x << y;

} // end function operator<<

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