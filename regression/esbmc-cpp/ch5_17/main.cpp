// Fig. 5.24: fig05_24.cpp
// Card shuffling dealing program.
#include <iostream>

using std::cout;
using std::left;
using std::right;

#include <iomanip>

using std::setw;

#include <cstdlib>  // prototypes for rand and srand
#include <ctime>    // prototype for time

// prototypes
void shuffle( int [][ 13 ] );
void deal( const int [][ 13 ], const char *[], const char *[] );

int main()
{
   // initialize suit array
   const char *suit[ 4 ] =  
      { "Hearts", "Diamonds", "Clubs", "Spades" };
   
   // initialize face array
   const char *face[ 13 ] = 
      { "Ace", "Deuce", "Three", "Four",
        "Five", "Six", "Seven", "Eight",
        "Nine", "Ten", "Jack", "Queen", "King" };

   // initialize deck array
   int deck[ 4 ][ 13 ] = { 0 };

   srand( time( 0 ) );        // seed random number generator

   shuffle( deck );
   deal( deck, face, suit );

   return 0;  // indicates successful termination

} // end main

// shuffle cards in deck
void shuffle( int wDeck[][ 13 ] )
{
   int row;
   int column;

   // for each of the 52 cards, choose slot of deck randomly
   for ( int card = 1; card <= 52; card++ ) {

      // choose new random location until unoccupied slot found
      do {
         row = rand() % 4;
         column = rand() % 13;
      } while( wDeck[ row ][ column ] != 0 ); // end do/while

      // place card number in chosen slot of deck
      wDeck[ row ][ column ] = card;

   } // end for

} // end function shuffle

// deal cards in deck
void deal( const int wDeck[][ 13 ], const char *wFace[],
           const char *wSuit[] )
{
   // for each of the 52 cards
   for ( int card = 1; card <= 52; card++ )

      // loop through rows of wDeck
      for ( int row = 0; row <= 3; row++ )

         // loop through columns of wDeck for current row
         for ( int column = 0; column <= 12; column++ )

            // if slot contains current card, display card
            if ( wDeck[ row ][ column ] == card ) {
               cout << setw( 5 ) << right << wFace[ column ] 
                    << " of " << setw( 8 ) << left 
                    << wSuit[ row ]
                    << ( card % 2 == 0 ? '\n' : '\t' );

            } // end if

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
