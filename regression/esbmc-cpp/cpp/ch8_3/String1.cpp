// Fig. 8.8: string1.cpp
// Member function definitions for class String.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

#include <new>        // C++ standard "new" operator

#include <cstring>    // strcpy and strcat prototypes
#include <cstdlib>    // exit prototype

#include "string1.h"  // String class definition

// conversion constructor converts char * to String
String::String( const char *s ) 
   : length( ( s != 0 ) ? strlen( s ) : 0 )
{
   cout << "Conversion constructor: " << s << '\n';
   setString( s );         // call utility function

} // end String conversion constructor

// copy constructor
String::String( const String &copy ) 
   : length( copy.length )
{
   cout << "Copy constructor: " << copy.sPtr << '\n';
   setString( copy.sPtr ); // call utility function

} // end String copy constructor

// Destructor
String::~String()
{
   cout << "Destructor: " << sPtr << '\n';
   delete [] sPtr;         // reclaim string

} // end ~String destructor

// overloaded = operator; avoids self assignment
const String &String::operator=( const String &right )
{
   cout << "operator= called\n";

   if ( &right != this ) {         // avoid self assignment
      delete [] sPtr;              // prevents memory leak
      length = right.length;       // new String length
      setString( right.sPtr );     // call utility function
   }

   else
      cout << "Attempted assignment of a String to itself\n";

   return *this;   // enables cascaded assignments

} // end function operator=

// concatenate right operand to this object and
// store in this object.
const String &String::operator+=( const String &right )
{
   size_t newLength = length + right.length;   // new length
   char *tempPtr = new char[ newLength + 1 ];  // create memory

   strcpy( tempPtr, sPtr );                 // copy sPtr
   strcpy( tempPtr + length, right.sPtr );  // copy right.sPtr

   delete [] sPtr;      // reclaim old space
   sPtr = tempPtr;      // assign new array to sPtr
   length = newLength;  // assign new length to length

   return *this;  // enables cascaded calls

} // end function operator+=

// is this String empty?
bool String::operator!() const 
{ 
   return length == 0; 

} // end function operator! 

// Is this String equal to right String?
bool String::operator==( const String &right ) const
{ 
   return strcmp( sPtr, right.sPtr ) == 0; 

} // end function operator==

// Is this String less than right String?
bool String::operator<( const String &right ) const
{ 
   return strcmp( sPtr, right.sPtr ) < 0; 

} // end function operator<

// return reference to character in String as lvalue
char &String::operator[]( int subscript )
{
   // test for subscript out of range
   if( subscript < 0 || subscript >= length ) {
      cout << "Error: Subscript " << subscript 
           << " out of range" << endl;

      exit( 1 );  // terminate program
   }

   return sPtr[ subscript ];  // creates lvalue

} // end function operator[]

// return reference to character in String as rvalue
const char &String::operator[]( int subscript ) const
{
   // test for subscript out of range
   if( subscript < 0 || subscript >= length ) {
      cout << "Error: Subscript " << subscript 
           << " out of range" << endl;

      exit( 1 );  // terminate program
   }

   return sPtr[ subscript ];  // creates rvalue

} // end function operator[]

// return a substring beginning at index and
// of length subLength
String String::operator()( int index, int subLength )
{
   // if index is out of range or substring length < 0, 
   // return an empty String object
   if ( index < 0 || index >= length || subLength < 0 )  
      return "";  // converted to a String object automatically

   // determine length of substring
   int len;

   if ( ( subLength == 0 ) || ( index + subLength > length ) )
      len = length - index;
   else
      len = subLength;

   // allocate temporary array for substring and 
   // terminating null character
   char *tempPtr = new char[ len + 1 ];

   // copy substring into char array and terminate string
   strncpy( tempPtr, &sPtr[ index ], len );
   tempPtr[ len ] = '\0';

   // create temporary String object containing the substring
   String tempString( tempPtr );
   delete [] tempPtr;  // delete temporary array

   return tempString;  // return copy of the temporary String

} // end function operator()

// return string length
int String::getLength() const 
{ 
   return length; 

} // end function getLenth

// utility function called by constructors and operator=
void String::setString( const char *string2 )
{
   sPtr = new char[ length + 1 ]; // allocate memory

   // if string2 is not a null pointer, copy contents 
   if ( string2 != 0 )
      strcpy( sPtr, string2 );    // copy literal to object

   // if string2 is a null pointer, make this an empty string
   else
      sPtr[ 0 ] = '\0';           // empty string

} // end function setString 

// overloaded output operator
ostream &operator<<( ostream &output, const String &s )
{
   output << s.sPtr;

   return output;   // enables cascading

} // end function operator<<

// overloaded input operator
istream &operator>>( istream &input, String &s )
{
   char temp[ 100 ];   // buffer to store input

   input >> setw( 100 ) >> temp;
   s = temp;        // use String class assignment operator

   return input;    // enables cascading

} // end function operator>>

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