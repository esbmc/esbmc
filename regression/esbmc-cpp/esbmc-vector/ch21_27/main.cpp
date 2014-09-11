// Fig. 21.42: fig21_42.cpp
// Demonstrating function objects.
#include <iostream>
#include <iterator>
#include <string>
using std::cout;
using std::endl;

#include <vector>      // vector class-template definition
#include <algorithm>   // copy algorithm
#include <numeric>     // accumulate algorithm
#include <functional>  // binary_function definition

// binary function adds square of its second argument and
// running total in its first argument, then returns sum
int sumSquares( int total, int value ) 
{ 
   return total + value * value; 

} // end function sumSquares

// binary function class template defines overloaded operator()
// that adds suare of its second argument and running total in 
// its first argument, then returns sum
template< class T > 
class SumSquaresClass : public std::binary_function< T, T, T > {

public:
   
   // add square of value to total and return result
   const T operator()( const T &total, const T &value )
   { 
      return total + value * value; 

   } // end function operator()
   
}; // end class SumSquaresClass

int main()
{
   const int SIZE = 10;
   int array[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

   std::vector< int > integers( array, array + SIZE );

   std::ostream_iterator< int > output( cout, " " );

   int result = 0;

   cout << "vector v contains:\n";
   std::copy( integers.begin(), integers.end(), output );

   // calculate sum of squares of elements of vector integers
   // using binary function sumSquares
   result = std::accumulate( integers.begin(), integers.end(), 
      0, sumSquares );

   cout << "\n\nSum of squares of elements in integers using "
        << "binary\nfunction sumSquares: " << result;

   // calculate sum of squares of elements of vector integers
   // using binary-function object 
   result = std::accumulate( integers.begin(), integers.end(), 
      0, SumSquaresClass< int >() );

   cout << "\n\nSum of squares of elements in integers using "
        << "binary\nfunction object of type " 
        << "SumSquaresClass< int >: " << result << endl;

   return 0;

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
