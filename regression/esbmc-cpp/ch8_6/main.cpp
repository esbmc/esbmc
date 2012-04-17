// Fig. 8.14: fig08_14.cpp
// Demonstrating standard library class vector.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include <iomanip>

using std::setw;

#include <vector>

using std::vector;
using namespace std;

void outputVector( const vector< int > & );
void inputVector( vector< int > & );

int main()
{
   vector< int > integers1;   // 7-element vector< int >
   vector< int > integers2;  // 10-element vector< int >

   // print integers1 size and contents
   cout << "Size of vector integers1 is " 
        << integers1.size()
        << "\nvector after initialization:\n";
   outputVector( integers1 );

   // print integers2 size and contents
   cout << "\nSize of vector integers2 is " 
        << integers2.size()
        << "\nvector after initialization:\n";
   outputVector( integers2 );

   // input and print integers1 and integers2
   cout << "\nInput 17 integers:\n";
   inputVector( integers1 );
   inputVector( integers2 );

   cout << "\nAfter input, the vectors contain:\n"
        << "integers1:\n";
   outputVector( integers1 );
   cout << "integers2:\n";
   outputVector( integers2 );

   // use overloaded inequality (!=) operator
   cout << "\nEvaluating: integers1 != integers2\n";

   if ( integers1 != integers2 )
      cout << "integers1 and integers2 are not equal\n";

   // create vector integers3 using integers1 as an
   // initializer; print size and contents
   vector< int > integers3( integers1 );  // copy constructor

   cout << "\nSize of vector integers3 is "
        << integers3.size()
        << "\nvector after initialization:\n";

   outputVector( integers3 );


   // use overloaded assignment (=) operator
   cout << "\nAssigning integers2 to integers1:\n";
   integers1 = integers2;

   cout << "integers1:\n";
   outputVector( integers1 );
   cout << "integers2:\n";
   outputVector( integers1 );

   // use overloaded equality (==) operator
   cout << "\nEvaluating: integers1 == integers2\n";

   if ( integers1 == integers2 )
      cout << "integers1 and integers2 are equal\n";

   // use overloaded subscript operator to create rvalue
   cout << "\nintegers1[5] is " << integers1[ 5 ];

   // use overloaded subscript operator to create lvalue
   cout << "\n\nAssigning 1000 to integers1[5]\n";
   integers1[ 5 ] = 1000;
   cout << "integers1:\n";
   outputVector( integers1 );

   // attempt to use out of range subscript
   cout << "\nAttempt to assign 1000 to integers1.at( 15 )"
        << endl;
   integers1.at( 15 ) = 1000;  // ERROR: out of range

   return 0;

} // end main

// output vector contents
void outputVector( const vector< int > &array )
{
	int i;
   for ( int i = 0; i < array.size(); i++ ) {
      cout << setw( 12 ) << array[ i ];

      if ( ( i + 1 ) % 4 == 0 ) // 4 numbers per row of output
         cout << endl;

   } // end for

   if ( i % 4 != 0 )
      cout << endl;

} // end function outputVector

// input vector contents
void inputVector( vector< int > &array )
{
   for ( int i = 0; i < array.size(); i++ )
      cin >> array[ i ];

} // end function inputVector



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
