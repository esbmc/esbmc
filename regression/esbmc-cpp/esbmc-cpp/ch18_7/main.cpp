// Fig. 18.14: fig18_14.cpp
// Representing cards with bit fields in a struct.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

// BitCard structure definition with bit fields
struct BitCard {
   unsigned face : 4;   // 4 bits; 0-15
   unsigned suit : 2;   // 2 bits; 0-3
   unsigned color : 1;  // 1 bit; 0-1

}; // end struct BitBard

void fillDeck( BitCard * const );    // prototype
void deal( const BitCard * const );  // prototype

int main()
{
   BitCard deck[ 52 ];

   fillDeck( deck );
   deal( deck );

   return 0;

} // end main

// initialize BitCards
void fillDeck( BitCard * const wDeck )
{
   for ( int i = 0; i <= 51; i++ ) {
      wDeck[ i ].face = i % 13;
      wDeck[ i ].suit = i / 13;
      wDeck[ i ].color = i / 26;

   } // end for

} // end function fillDeck

// output cards in two column format; cards 0-25 subscripted 
// with k1 (column 1); cards 26-51 subscripted k2 (column 2)
void deal( const BitCard * const wDeck )
{
   for ( int k1 = 0, k2 = k1 + 26; k1 <= 25; k1++, k2++ ) {
      cout << "Card:" << setw( 3 ) << wDeck[ k1 ].face 
           << "  Suit:" << setw( 2 ) << wDeck[ k1 ].suit 
           << "  Color:" << setw( 2 ) << wDeck[ k1 ].color 
           << "   " << "Card:" << setw( 3 ) << wDeck[ k2 ].face
           << "  Suit:" << setw( 2 ) << wDeck[ k2 ].suit 
           << "  Color:" << setw( 2 ) << wDeck[ k2 ].color 
           << endl;

   } // end for

} // end function deal

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