// Fig. 21.40: fig21_40.cpp                                
// Using a bitset to demonstrate the Sieve of Eratosthenes.
#include <iostream>

using std::cin;
using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

#include <bitset>  // bitset class definition
#include <cmath>   // sqrt prototype

int main()
{
   const int size = 1024;
   int value;
   std::bitset< size > sieve;

   sieve.flip();

   // perform Sieve of Eratosthenes
   int finalBit = sqrt( sieve.size() ) + 1;

   for ( int i = 2; i < finalBit; ++i ) 

      if ( sieve.test( i ) ) 

         for ( int j = 2 * i; j < size; j += i ) 
            sieve.reset( j );

   cout << "The prime numbers in the range 2 to 1023 are:\n";

   // display prime numbers in range 2-1023
   for ( int k = 2, counter = 0; k < size; ++k )

      if ( sieve.test( k ) ) {
         cout << setw( 5 ) << k;

         if ( ++counter % 12 == 0 ) 
            cout << '\n';

      } // end outer if          
   
   cout << endl;

   // get value from user to determine whether value is prime
   cout << "\nEnter a value from 1 to 1023 (-1 to end): ";
   cin >> value;

   while ( value != -1 ) {

      if ( sieve[ value ] )
         cout << value << " is a prime number\n";
      else
         cout << value << " is not a prime number\n";
      
      cout << "\nEnter a value from 2 to 1023 (-1 to end): ";
      cin >> value;

   } // end while

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