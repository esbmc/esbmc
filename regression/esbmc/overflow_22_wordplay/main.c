/*

Wordplay Version 7.22         03-20-96

Written by Evans A Criswell at the University of Alabama in Huntsville

03-20-96 Fixed a small memory allocation problem.  In a couple of places,
	 the amount allocated to hold character strings was not taking the
	 space to store the null into account.  This bug has only affected
	 a couple of people.  
09-11-95 In the anagramr7 function, I check the product of the maximum
	 "levels deep" remaining and the length of the longest candidate
	 word.  If this product is less than the length of the string
	 passed in, a "dead end" condition exists.  This makes the program
	 run significantly faster for longer strings if the maximum
	 depth option is used.
08-21-94 Added "wordfile from stdin" option using "-f -"  
	 Fixed "4" bug.  Digits in a string disqualify the string.
	 Vowel-check override option added.
	 Starting word ("w" option) checked to see if it's an anagram
	 of the initial string.
08-16-94 Used integer masks representing which letters appear in each
	 word, allowing extraction checking to be checked quickly for
	 failure in the anagramr7 routine.  Result:  the program has
	 been 4 to 5 times faster.
08-14-94 Made the program much more memory efficient.  Instead of calling
	 malloc for each word in the candidate word list and in the key
	 list, a contiguous block of memory was allocated to hold the
	 words.  The block is realloc'ed if it needs to be increased as
	 the words are read in.  After the words are packed into the
	 block, the pointers are allocated and are pointed to the
	 appropriate places (beginnings of words) in the block, so the
	 rest of the program works with no modification.  Two gigantic
	 arrays that weren't being used were eliminated.  The word length
	 index arrays are now made to be the size of the longest word 
	 instead of MAX_WORDS.  In fact, MAX_WORDS is now obsolete.
07-14-94 Added "silent" option.
06-03-94 Added "#include <ctypes.h>" so it would work on BSD/386 .  Thanks
         to mcintyre@io.com (James Michael Stewart) for reporting the bug.
05-26-94 Fixed command-line parsing bug.
05-25-94 Eliminated redundant permutations.  Added option to specify a
	 word to appear in anagrams.  Added maximum depth option (number
	 of words, maximum, to appear in an anagram).  
05-24-94 Added option so user could specify whether to allow anagrams
	 with adjacent duplicate words like "A A" or "DOG DOG".
05-16-94 Made a second copy of the word list and sorted each word's
	 letters alphabetically and sorted this list of keys alphabetically.
	 Modified the recursive algorithm to use the new index. (Ver 6.00)
05-16-94 Another little bug fix.  Someone found that, on their machine,
	 if there are no candidate words loaded for the string being 
         anagrammed, it causes an error when malloc gets passed a zero
         value for the amount to allocate.
05-13-94 Tiny bug fix.  Just a small bug that never actually caused a
	 crash, but very well could have if it had wanted to.  :-)
04-25-94 Speed increase.  If exts indicates extraction was impossible,
	 continue (try next word) instead of executing rest of loop body.
04-21-91 Ron Gregory found a simple bug that has been in all the C
	 versions (4.00 through 5.20).  In the one-word anagram
	 section, a less than should have been a less than or equal to.
	 A simple fencepost error.  The recursive anagram procedure had
	 a similar problem.  A severe error was fixed in the version
	 5.20 read routine which caused the program not to read the
	 wordfile correctly if the entries were lowercase.
04-17-94 Since this program, since it was ported to C, is command-line
	 based, and only anagrams one string, it is not necessary to
	 store the wordlist internally.  Unnecessary words are weeded
	 out as the list is being read, using the "extract" routine.
	 I can't believe I didn't think of using that routine for that
	 purpose sooner.  That means pass1 and pass2 are obsolete.
04-14-94 Changed the "extract" function to use pointers instead of
	 array notation.  Under some compilers, this may nearly double
	 the execution speed of the recursive anagram procedure.  On
	 other compilers, it may make no difference at all.
04-11-94 Added the minimum and maximum candidate word length options
	 that were available in version 3.00 when the program was 
	 interactive.  This helps to narrow down the word list and
	 eliminate a lot of short words when anagramming long strings.
11-30-93 Fixed a bug that Versions 5.00 and 5.01 had.  If there were
	 no words in the candidate word list with the same length as
	 the string passed to anagramr, the string passed to anagramr
	 would not be anagrammed, causing many possible anagrams to
	 be missed.
11-08-93 Eliminated anagrams consisting of the same word occurring 
         multiple times in a row, such "IS IS ...", since interesting
         anagrams rarely contain such repetitions. (Version 5.01)
11-08-93 Debug print statements commented and output cleaned up.
         Version 5.00 completed.  It is currently not known which is
	 always faster:  the old iterative 2 and 3 word anagram options
	 or the recursive algorithm.  All the options from version 4.00
	 are still in the program.
11-07-93 Recursive algorithm working!
11-03-93 Added code to index the candidate word list by number of vowels
	 per word. (Beginning of 5.00 Alpha)  Never used in Version 5.00,
	 but the code is there for future use.
05-25-93 Three word anagramming capability ported and added.
04-30-93 The big port from FORTRAN 77 to ANSI C.  No longer interactive.
	 Instead, arguments are taken from the command line.
	 (Everything working except three-word anagrams and all command
	 line options not yet implemented)

Version 4.00 is the first version to be implemented in C.  All previous
versions were written in FORTRAN 77.

Note:  There was no version 5.12.  It was called 5.20 instead.

Version 7.22  03-20-96  Bug fix.
Version 7.21  09-11-95  Speed increase.
Version 7.20  08-21-94  Wordfile from stdin capability, bug fixes.
Version 7.11  08-16-94  Speed increase.
Version 7.10  08-14-94  Program uses much less memory.
Version 7.02  07-14-94  Silent option.
Version 7.01  06-03-94  Portability problem fixed.  ctypes.h needed .
Version 7.00  05-26-94  Redundant permutations eliminated. Several refinements.
Version 6.00  05-17-94  Huge speed increase.
Version 5.24  05-16-94  Bug fix.
Version 5.23  05-13-94  Tiny bug fix.
Version 5.22  04-25-94  Speed increase.
Version 5.21  04-21-94  Bug fixes.
Version 5.20  04-17-94  Faster program initialization.  Far less memory used.
Version 5.11  04-14-94  Slight speed increase with some compilers
Version 5.10  04-11-94  Minimum, maximum candidate word length again
			available.  (First time available in the C versions).
Version 5.02  11-30-93  Bug fix. 
Version 5.01  11-08-93  Optimization to eliminate multiple occurrences
                        of a particular word in a row.
Version 5.00  11-08-93  Recursive algorithm added  
Version 4.00  04-30-93  Ported to C.  Became non-interactive and more
			suitable for UNIX environment  
Version 3.00  12-16-91  Indexing improvements.  Huge speed increase 
Version 2.10  04-16-91  Options and help added 
Version 2.00  04-12-91  Three word anagrams added 
Version 1.11  04-11-91  Bug fixes and cleanups 
Version 1.10  04-03-91  Pass 2 word filter added.  Huge speed increase. 
Version 1.00  03-29-91  One and two word anagrams 

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define max(A, B) ((A) > (B) ? (A) : (B))
#define min(A, B) ((A) < (B) ? (A) : (B))

#define DEFAULT_WORD_FILE "/usr/share/games/wordplay/words721.txt"
#define WORDBLOCKSIZE 4096
#define MAX_WORD_LENGTH 128
#define SAFETY_ZONE MAX_WORD_LENGTH + 1
#define MAX_ANAGRAM_WORDS 32
#define MAX_PATH_LENGTH 256

char   *uppercase (char *s);
char   *alphabetic (char *s);
int     numvowels (char *s);
void    anagramr7 (char *s, char **accum, int *minkey, int *level);
char   *extract (char *s1, char *s2);
int     intmask (char *s);

char  **words2;  /* Candidate word index (pointers to the words) */
char   *words2mem;  /* Memory block for candidate words  */
char  **words2ptrs; /* For copying the word indexes */
char  **wordss;    /* Keys */
char   *keymem;     /* Memory block for keys */
int    *wordsn;    /* Lengths of each word in words2 */
int    *wordmasks; /* Mask of which letters are contained in each word */
int     ncount;    /* Number of candidate words */
int     longestlength; /*  Length of longest word in words2 array */
char    largestlet;
int     rec_anag_count;  /*  For recursive algorithm, keeps track of number
			 of anagrams fond */
int     adjacentdups;
int     specfirstword;
int     maxdepthspec;
int     silent;
int     max_depth;
int     vowelcheck;

int    *lindx1; 
int    *lindx2;
int     findx1[26];
int     findx2[26];

int main (int argc, char *argv[])
{
  FILE    *word_file_ptr;
  char     buffer[MAX_WORD_LENGTH];
  char     ubuffer[MAX_WORD_LENGTH]; 
  char     alphbuffer[MAX_WORD_LENGTH];
  char     initword[MAX_WORD_LENGTH];
  char     remaininitword[MAX_WORD_LENGTH];
  char     word_file_name[MAX_PATH_LENGTH];
  char     first_word[MAX_WORD_LENGTH];
  char     u_first_word[MAX_WORD_LENGTH];
  char     tempword[MAX_WORD_LENGTH];
  int      ilength;                           /* Length of initword */
  int      size; 
  int      gap;
  int      switches;
  int      iholdn;
  char     chold;
  char    *wholdptr;
  int      curlen;
  int      curpos; 
  char     curlet; 
  int      icurlet;
  int      recursiveanag;
  int      listcandwords; 
  int      wordfilespec;
  int      firstwordspec; 
  int      maxcwordlength; 
  int      mincwordlength;
  int      iarg;
  int      keyi; 
  int      keyj;
  char   **accum;
  int      level;
  int      minkey;
  char     leftover[MAX_WORD_LENGTH];
  int      w2size;
  char    *w2memptr;
  int      w2offset;
  char    *keymemptr;
  int      keyoffset;
  char     no[3] = "no";
  char     yes[4] = "yes";
  int      fileinput;
  int      hasnumber;
  int      i; 
  int      j; 
  int      k;

/*
  printf ("Command line parameters:\n");
  for (i = 0; i < argc; i++) printf ("\"%s\" ", argv[i]);
  printf ("\n");
*/

  if (argc < 2)
  {
    fprintf (stderr, 
	    "Wordplay Version 7.22  03-20-96, 1991   by Evans A Criswell\n");
    fprintf (stderr, 
	    "University of Alabama in Huntsville     criswell@cs.uah.edu\n\n");
    fprintf (stderr, "Usage:  ");
    fprintf (stderr, "wordplay string_to_anagram [-slxavnXmXdX] [-w word] "
		     "[-f word_file]\n\n");
    fprintf (stderr, "Capital X represents an integer.\n\n");
    fprintf (stderr, "s  = silent operation (no header or line numbers)\n");
    fprintf (stderr, "l  = print candidate word list\n");
    fprintf (stderr, "x  = do not generate anagrams (useful with l option)\n");
    fprintf (stderr, "a  = multiple occurrences of a word in an anagram OK\n");
    fprintf (stderr, "v  = allow words with no vowels to be considered\n");
    fprintf (stderr, "nX = candidate words must have n characters minimum\n");
    fprintf (stderr, "mX = candidate words must have m characters maximum\n");
    fprintf (stderr, "dX = limit anagrams to d words\n\n");
    fprintf (stderr, "w word = word to start anagrams\n");
    fprintf (stderr, "f file = word file to use (\"-f -\" for stdin)\n\n");
    fprintf (stderr, "Suggestion:  Run \"wordplay trymenow\" "
		     " to get started.\n");
    exit(-1);
  }

  strcpy (word_file_name, DEFAULT_WORD_FILE);

  recursiveanag = 1;
  listcandwords = 0;
  wordfilespec = 0;
  firstwordspec = 0;
  specfirstword = 0;     /*  this is the permanent one */
  silent = 0;
  vowelcheck = 1;
  maxdepthspec = 0;

  maxcwordlength = MAX_WORD_LENGTH;
  mincwordlength = 0;

  max_depth = MAX_ANAGRAM_WORDS;

  iarg = 1;   
  while (iarg < argc)
  {
    if (wordfilespec == 1)
    {
      strcpy (word_file_name, argv[iarg]);
      iarg++;
      wordfilespec = 0;
      continue;
    }
    if (firstwordspec == 1)
    {
      strcpy (first_word, argv[iarg]);
      iarg++;
      firstwordspec = 0;
      continue;
    }
    if (argv[iarg][0] == '-')
    {
      if ((int) strlen(argv[iarg]) > 1)
      {
	i = 1;
	while (i < (int) strlen(argv[iarg]))
	{
	  switch (argv[iarg][i])
	  {
            case 'a' : adjacentdups = 1;
		       break;
	    case 'l' : listcandwords = 1;
		       break;
	    case 'f' : wordfilespec = 1;
		       break;
            case 'x' : recursiveanag = 0;
		       break;
            case 's' : silent = 1;
		       break;
            case 'v' : vowelcheck = 0;
		       break;
            case 'w' : firstwordspec = 1;
		       specfirstword = 1;
		       break;
            case 'm' : maxcwordlength = 0;
		       i++;
		       while ((argv[iarg][i] >= '0') && (argv[iarg][i] <= '9'))
			 maxcwordlength = maxcwordlength * 10 + 
					  ((int) argv[iarg][i++] - (int) '0');
                       i--;
		       break;
            case 'n' : i++;
		       while ((argv[iarg][i] >= '0') && (argv[iarg][i] <= '9'))
			 mincwordlength = mincwordlength * 10 +
					  ((int) argv[iarg][i++] - (int) '0');
                       i--;
		       break;
            case 'd' : maxdepthspec = 1;
                       max_depth = 0;
		       i++;
		       while ((argv[iarg][i] >= '0') && (argv[iarg][i] <= '9'))
			 max_depth = max_depth * 10 +
			             ((int) argv[iarg][i++] - (int) '0');
                       i--;
		       break;
            default  : fprintf (stderr, "Invalid option: \"%c\" - Ignored\n", 
				argv[iarg][i]);
                       break;
	  }
	  i++;
	}
      }
      iarg++;
    }
    else
    {
      strcpy (initword, uppercase(argv[iarg]));
      iarg++;
    }
  }

  if (silent == 0)
  {
    printf ("Wordplay Version 7.22  03-20-96, 1991   by Evans A Criswell\n");
    printf ("University of Alabama in Huntsville     criswell@cs.uah.edu\n\n");
  }

  if (silent == 0)
  {
    printf ("\n");
    printf ("Candidate word list :  %s\n", (listcandwords == 0) ? no : yes);
    printf ("Anagram Generation  :  %s\n", (recursiveanag == 0) ? no : yes);
    printf ("Adjacent duplicates :  %s\n", (adjacentdups == 0) ? no : yes);
    printf ("Vowel-free words OK :  %s\n\n", (vowelcheck == 0) ? yes : no);

    printf ("Max anagram depth   :  %d\n", max_depth);
    printf ("Maximum word length :  %d\n", maxcwordlength);
    printf ("Minimum word length :  %d\n\n", mincwordlength);

    if (specfirstword)
      printf ("First word          :  \"%s\"\n", first_word);

    printf ("Word list file      :  \"%s\"\n", word_file_name);
    printf ("String to anagram   :  \"%s\"\n", initword);
    printf ("\n");
  }

/* Remove non-alphabetic characters from initword */

  strcpy (tempword, alphabetic (initword));
  strcpy (initword, tempword);

  ilength = (int) strlen (initword);

/*  Sort characters of initword in increasing order  */

  size = ilength;
  gap = size;
  do
  {
    gap = max (((gap * 10) / 13), 1);
    switches = 0;
    for (i = 0; i < (size - gap); i++)
    {
      j = i + gap;
      if (initword[i] > initword[j])
      {
	chold = initword[i];
	initword[i] = initword[j];
	initword[j] = chold;
	switches++;
      }
    }
  }
  while ((switches != 0) | (gap != 1));

/*  Extract first_word (if specified) from initword and store in
    remaininitword  */

  if (specfirstword)
  {
    strcpy (u_first_word, uppercase(first_word));
    strcpy (remaininitword, extract (initword, u_first_word));
    if (remaininitword[0] == '0')
    {
      fprintf (stderr, "Specified first word \"%s\" cannot be extracted "
		       "from initial string \"%s\"\n", u_first_word, initword);
      exit (1);
    }
    if (strlen (remaininitword) == 0)
    {
      if (silent == 0) 
      {
	printf ("Anagrams found:\n");
	printf ("     0.  %s\n", u_first_word);
      }
      else
	printf ("%s\n", u_first_word);
      exit (0);
    }

  }

/*  Allocate memory for the words themselves  */

  w2size = WORDBLOCKSIZE;

  if ((words2mem = (char *) malloc (w2size * sizeof (char))) == (char *) NULL)
  {
    fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
    exit (-1);
  }

/* Open the word file and read the words. */

  if (silent == 0)
  {
    printf ("\nInitializing.  Please wait while words are being loaded\n");
    printf ("and unnecessary words are being filtered out ...\n");
  }

  if (strcmp(word_file_name, "-") == 0)
  {
    fileinput = 0;
    word_file_ptr = stdin;
  }
  else 
  {
    if ((word_file_ptr = fopen (word_file_name, "r")) == NULL)
    {
      fileinput = 1;
      fprintf (stderr, "Error opening word file.\n");
      return (-1);
    }
  }

  i = 0;
  w2memptr = words2mem;
  w2offset = 0;
  longestlength = 0;

  while (fgets (buffer, MAX_WORD_LENGTH, word_file_ptr) != 
	 (char *) NULL)
  {
    j = (int) strlen (buffer) - 1;

/*  Replace the newline with a null  */

    buffer[j--] = '\0';

    strcpy (alphbuffer, alphabetic (buffer));

    if (((int) strlen (alphbuffer) < mincwordlength) || 
	((int) strlen (alphbuffer) > maxcwordlength))
      continue;

    hasnumber = 0;
    for (j = 0; j < (int) strlen (buffer); j++)
      if ((buffer[j] >= '0') && (buffer[j] <= '9')) hasnumber = 1;

    if (hasnumber == 1) continue;

    strcpy (ubuffer, uppercase (alphbuffer));
    strcpy (leftover, extract (initword, ubuffer));
    if (leftover[0] == '0') continue;

    strcpy (w2memptr, uppercase(buffer));
    w2memptr += strlen (buffer) + 1;
    w2offset += strlen (buffer) + 1;

    if ((int) strlen (alphbuffer) > longestlength) 
      longestlength = strlen (alphbuffer);

    if ((w2size - w2offset) < SAFETY_ZONE) 
    {
       w2size += WORDBLOCKSIZE;
       if ((words2mem = (char *) realloc (words2mem, w2size)) == (char *) NULL)
       {
         fprintf (stderr, "Out of memory.  realloc() returned NULL.\n");
         exit (-1);
       }
       w2memptr = words2mem + w2offset;
    }

    i++; 
    ncount = i;
  }

  if (fileinput == 1) fclose (word_file_ptr);

/* Malloc pointers for the word indexes */

  if ((words2 = (char **) malloc (ncount * sizeof (char *))) == (char **) NULL)
  {
    fprintf (stderr, "Insufficient memory.  malloc() returned NULL.\n");
    exit (-1);
  }

/*  Go through the loaded words and index the beginning of each word */

  words2[0] = words2mem;
  j = 1;
  for (i = 0; i < w2size; i++)
    if (j < ncount)
      if (words2mem[i] == '\0') words2[j++] = words2mem + i + 1;


  if (silent == 0) printf ("\n%d words loaded (%d byte block).  " 
                           "Longest kept:  %d letters.\n",
			    ncount, w2size, longestlength);

  if (ncount == 0)
  {
    if (silent == 0)
      printf ("\nNo candidate words were found, so there are no anagrams.\n");
    exit(0);
  }

/* Store lengths of words from words2 array in wordsn array */

  if ((wordsn = (int *) malloc (ncount * sizeof (int))) == NULL)
  {
    fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
    exit (-1);
  }

  for (i = 0; i < ncount; i++) 
  {
    strcpy (alphbuffer, alphabetic (words2[i]));
    wordsn[i] = (int) strlen (alphbuffer); 
  }

/* Make a copy of the pointers from the words2 array (called words2ptrs) */

  if ((words2ptrs = (char **) malloc (ncount * sizeof (char *))) == 
      (char **) NULL)
  {
    fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
    exit (-1);
  }

  for (i = 0; i < ncount; i++) words2ptrs[i] = words2[i];

/* Make a copy of the word list, then sort each word in the new list 
   putting letters of the words in alphabetical order */

/*  Malloc the pointers for the list of keys */

  if ((wordss = (char **) malloc (ncount * sizeof (char *))) == (char **) NULL)
  {
    fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
    exit (-1);
  }

/*  Make a copy of the block of memory containing the candidate word list */

  if ((keymem = (char *) malloc (w2size * sizeof (char))) == (char *) NULL)
  {
    fprintf (stderr, "Insufficient memory; malloc() returned NULL.\n");
    exit (-1);
  }

/*  Copy the words from the candidate word block, one by one, eliminating
    non-alphabetic characters. */

  keymemptr = keymem;
  keyoffset = 0;

  for (i = 0; i < ncount; i++)
  {
    strcpy (alphbuffer, alphabetic (words2[i]));
    strcpy (ubuffer, uppercase (alphbuffer));
    strcpy (keymemptr, ubuffer);
    keymemptr += wordsn[i] + 1;
    keyoffset += wordsn[i] + 1;

  }

/*  Setup the pointers to the beginnings of the words, as we did earlier
    for the candidate word indexes */

  wordss[0] = keymem;
  j = 1;
  for (i = 0; i < w2size; i++)
    if (j < ncount)
      if (keymem[i] == '\0') wordss[j++] = keymem + i + 1;

/*  Create the keys by sorting the characters of the words in the keymem space 
    in place, using the wordss index pointers.  */

  for (k = 0; k < ncount; k++)
  {
    size = (int) strlen (wordss[k]);
    gap = size;
    do
    {
      gap = max (((gap * 10) / 13), 1);
      switches = 0;
      for (i = 0; i < (size - gap); i++)
      {
	j = i + gap;
	if (wordss[k][i] > wordss[k][j])
	{
	  chold = wordss[k][i];
	  wordss[k][i] = wordss[k][j];
	  wordss[k][j] = chold;
	  switches++;
        }
      }
    }
    while ((switches != 0) | (gap != 1));
  }

/* Sort the second "sorted" list of candidate words by first letter, 
   keeping references to the original word list, sorted by length (words2)
   intact (the words2ptrs array).   */

  size = ncount;
  gap = size;
  do
  {
    gap = max (((gap * 10) / 13), 1);
    switches = 0;
    for (i = 0; i < (size - gap); i++)
    {
      j = i + gap;
      if (strcmp (wordss[i], wordss[j]) > 0)
      {
	wholdptr = wordss[i];
	wordss[i] = wordss[j];
	wordss[j] = wholdptr;
	wholdptr = words2ptrs[i];
	words2ptrs[i] = words2ptrs[j];
	words2ptrs[j] = wholdptr;
	switches++;
      }
    }
  }
  while ((switches != 0) | (gap != 1));
  largestlet = wordss[ncount - 1][0];

/* Sort the list of candidate words (words2 array) by length */

  size = ncount;
  gap = size;
  do
  {
    gap = max (((gap * 10) / 13), 1);
    switches = 0;
    for (i = 0; i < (size - gap); i++)
    {
      j = i + gap;
      keyi = wordsn[i];
      keyj = wordsn[j];
      if (keyi > keyj)
      {
        iholdn = wordsn[i];
        wordsn[i] = wordsn[j];
        wordsn[j] = iholdn;
        wholdptr = words2[i];
        words2[i] = words2[j];
        words2[j] = wholdptr;
        switches++;
      }
    }
  }
  while ((switches != 0) | (gap != 1));

/* Print candidate word list */

  if (listcandwords)
  {
    if (silent == 0) printf ("\nList of candidate words:\n");
    for (i = 0; i < ncount; i++) 
    {
      if (silent == 0) 
	printf ("%6d.  %s\n", i, words2[i]);
      else
	printf ("%s\n", words2[i]);
    }
  }


/* Create indexes into words2 array by word length.  Words of length i
   will be in elements lindx1[i] through lindx2[i] of array words2.
   Of course, the algorithm below works because words2 has already
   been sorted by word length earlier.  */

  if ((lindx1 = (int *) malloc ((longestlength + 1) * sizeof (int))) 
	                == (int *) NULL) 
  {
    fprintf (stderr, "Insufficient memory.  malloc() returned NULL.\n");
    exit (-1);
  }

  if ((lindx2 = (int *) malloc ((longestlength + 1) * sizeof (int))) 
	                == (int *) NULL) 
  {
    fprintf (stderr, "Insufficient memory.  malloc() returned NULL.\n");
    exit (-1);
  }

  for (i = 0; i <= longestlength; i++)
  {
    lindx1[i] = -1;
    lindx2[i] = -2;
  }

  if (ncount > 0)
  {
    curpos = 0;
    curlen = wordsn[curpos];
    lindx1[curlen] = curpos;
    do
    {
      while (curpos < ncount)
      {
	if (wordsn[curpos] == curlen)
	  curpos++;
        else
	  break;
      }

      if (curpos >= ncount)
      {
        lindx2[curlen] = ncount - 1;
        break;
      }
      lindx2[curlen] = curpos - 1;
      curlen = wordsn[curpos];
      lindx1[curlen] = curpos;
    }
    while (curpos < ncount); 
  }

/* Create indexes into wordss array by first letter.  Words with first
   letter "A" will be will be in elements findx1[i] through findx2[i] 
   of array wordss.  Of course, the algorithm below works because 
   wordss has already been sorted by first letter earlier.  */

/*
  printf ("Beginning creation of first letter indexes.\n");
*/

  for (i = 0; i < 26; i++)
  {
    findx1[i] = -1;
    findx2[i] = -2;
  }

  if (ncount > 0)
  {
    curpos = 0;
    curlet = wordss[curpos][0];
    icurlet = (int) curlet - (int) 'A';
    findx1[icurlet] = curpos;
    do
    {
      while (curpos < ncount)
      {
	if (wordss[curpos][0] == curlet) 
	  curpos++;
        else
	  break;
      }
      if (curpos >= ncount)
      {
        findx2[icurlet] = ncount - 1;
        break;
      }
      findx2[icurlet] = curpos - 1;
      curlet = wordss[curpos][0];
      icurlet = (int) curlet - (int) 'A';
      findx1[icurlet] = curpos;
    }
    while (curpos < ncount); 
  }

/* Create masks (integers describing which letters are in each word */

  if ((wordmasks = (int *) malloc (ncount * sizeof (int))) == NULL)
  {
    fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
    exit (-1);
  }

  for (i = 0; i < ncount; i++) wordmasks[i] = intmask (wordss[i]);

/* Do recursive method of finding anagrams */

  if ((specfirstword == 0) && (recursiveanag))
  {
    if (silent == 0) printf ("\nAnagrams found:\n");

    if ((accum = (char **) malloc (MAX_ANAGRAM_WORDS * sizeof (char *))) == 
	(char **) NULL)
    {
      fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
      exit(-1);
    }

    for (i = 0; i < MAX_ANAGRAM_WORDS; i++)
      if ((accum[i] = (char *) malloc ((longestlength + 1) * sizeof (char))) ==
	  (char *) NULL)
      {
	fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
	exit(-1);
      }

    accum[0][0] = '\0';
    level = 0;
    rec_anag_count = 0;

    minkey = findx1[(int) initword[0] - (int) 'A'];

    anagramr7 (initword, accum, &minkey, &level);
    if (rec_anag_count == 0) 
      if (silent == 0) 
	printf ("\nNo anagrams found by recursive algorithm.\n");
  }

  if ((specfirstword == 1) && (recursiveanag))
  {
    if (silent == 0) printf ("\nRecursive anagrams found:\n");

    if ((accum = (char **) malloc (MAX_ANAGRAM_WORDS * sizeof (char *))) ==
	(char **) NULL)
    {
      fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
      exit(-1);
    }

    for (i = 0; i < MAX_ANAGRAM_WORDS; i++)
      if ((accum[i] = (char *) malloc ((MAX_WORD_LENGTH + 1) * sizeof (char))) ==
	  (char *) NULL)
      {
	fprintf (stderr, "Insufficient memory; malloc returned NULL.\n");
	exit(-1);
      }

    strcpy (accum[0], u_first_word);
    level = 1;
    rec_anag_count = 0;

    minkey = findx1[(int) remaininitword[0] - (int) 'A'];

    anagramr7 (remaininitword, accum, &minkey, &level);
    if (rec_anag_count == 0)
      printf ("\nNo anagrams found by recursive algorithm.\n");

  }
  return(0);
}

char *uppercase (char *s)
{
  static char upcasestr[MAX_WORD_LENGTH + 1];
  int i;

  for (i = 0; i < (int) strlen (s); i++) upcasestr[i] = toupper(s[i]);
  upcasestr[i] = '\0';

  return (upcasestr);
}

char *alphabetic (char *s)
{
  static char alphstr[MAX_WORD_LENGTH + 1];
  int i, pos;

  pos = 0;
  for (i = 0; i < (int) strlen (s); i++) 
    if (((s[i] >= 'A') && (s[i] <= 'Z')) || ((s[i] >= 'a') && (s[i] <= 'z')))
      alphstr[pos++] = s[i];
  alphstr[pos] = '\0';

  return (alphstr);
}

int numvowels (char *s)
{
  int vcount;
  char *cptr;

  vcount = 0;

  for (cptr = s; *cptr != '\0'; cptr++)
    switch (*cptr)
    {
      case 'A':  case 'E':  case 'I':  case 'O':  case 'U':  case 'Y':  
	vcount++; break;
    }
  return (vcount);
}

void anagramr7 (char *s, char **accum, int *minkey, int *level)
{
  int i, j, extsuccess, icurlet, newminkey, s_mask;
  char exts[MAX_WORD_LENGTH];

/*  Print arguments passed in for debugging purposes */

/*
  printf ("------------------------------------------------\n");
  printf ("anagramr called with: (\"%s\", (", s);  

  for (i = 0; i < *level; i++) printf ("\"%s\" ", accum[i]);
  printf ("), %d, %d)\n", *minkey, *level);
*/

/*  Exceeded depth specified by user */

  if (*level >= max_depth) 
  {
    (*level)--;
    return;
  }

/*  If the number of allowable additional "levels" times the length of
    the longest candidate word is less than the length of the string
    passed in, we know this is a "dead end".    */

  if (maxdepthspec == 1)
    if ((max_depth - *level) * longestlength < strlen(s))
    {
      (*level)--;
      return;
    }

/*  If no vowels, dead end  */

  if (vowelcheck == 1)
    if (numvowels (s) == 0)
    {
      (*level)--;
      return;
    }

  s_mask = intmask (s);

/*  Try to extract words and recursively apply algortihm  */

  extsuccess = 0;

  icurlet = (int) s[0] - (int) 'A';
  for (i = max (*minkey, findx1[icurlet]); i <= findx2[icurlet]; i++)
  {

/*
    printf ("Considering word \"%s\" (key \"%s\").  s = \"%s\" and i = %d\n",
	     words2ptrs[i], wordss[i], s, i);
*/

/*  Quick check for extraction.  If it fails, the extract check will fail.
    If this one passes, we must still do the extract a few steps below.  */

    if ((s_mask | wordmasks[i]) != s_mask) continue;

/*  Word used twice in a row in accumulation -- most likely not a meaningful
    anagram -- treat as a dead end  */

    if (adjacentdups == 0)
      if ((*level > 0) && (strcmp (words2ptrs[i], accum[*level - 1]) == 0)) 
        continue;

/*  Extract a word from the string being anagrammed.  */

    strcpy (exts, extract (s, wordss[i]));

/*  If the extraction was not possible, we are at a "dead end"  */

    if (*exts == '0') continue;

/*  If the extraction was perfect (left no letters), we've found
    an anagram.   */

    if (*exts == '\0')
    {
      rec_anag_count++;
      strcpy (accum[*level], words2ptrs[i]);
      if (silent == 0) printf ("%6d.  ", rec_anag_count);
      for (j = 0; j < *level; j++) printf ("%s ", accum[j]);
      printf ("%s\n", words2ptrs[i]);
      extsuccess = 1;
      continue;
    }

/*  The extraction was successful, but we must recursively call
    the procedure on what is left.  */

    extsuccess = 1;

    strcpy (accum[*level], words2ptrs[i]);
    (*level)++;

    if (adjacentdups == 0)
      newminkey = i + 1;
    else
      newminkey = i;

    anagramr7 (exts, accum, &newminkey, level);
  }

/*  Check to see if no extractions were a success */

  if (extsuccess == 0)
  {
    (*level)--;
    return;
  }
  (*level)--;

  return;
}

char *extract (char *s1, char *s2)
{

/*  Returns the characters remaining in s1 after extracting the characters
    one by one that appear in s2.  If the extraction is impossible (if s2
    contains a character not in s1), the string "0" (zero) is returned.   If 
    no characters remain in s1 after the extraction, then the null string ""
    is returned.

    Examples:  extract ("STOP", "SO")  returns "TP"
               extract ("AAA", "A") returns "AA"
               extract ("BCA", "ABC") returns ""
               extract ("ABCDE", "ABF") returns "0"  ('zero', not 'oh')
*/

  static char r1[MAX_WORD_LENGTH];
  char t1[MAX_WORD_LENGTH];
  char *s1p, *s2p, *r1p, *s1end, *s2end;
  int found, s1len, s2len;

  r1p = r1;

  strcpy (t1, s1);
  s1p = t1;
  s1len = (int) strlen (s1p);
  s1end = s1p + s1len;

  s2p = s2;
  s2len = (int) strlen (s2);
  s2end = s2p + s2len;

  for (s2p = s2; s2p < s2end; s2p++)
  {
    found = 0;
    for (s1p = t1; s1p < s1end; s1p++)
    {
      if (*s2p == *s1p)
      {
        *s1p = '0';
        found = 1;
        break;
      }
    }
    if (found == 0) 
    {
      *r1 = '0';
      *(r1 + 1) = '\0';
      return (r1);
    }
  }

  r1p = r1;
  for (s1p = t1; s1p < s1end; s1p++)
    if (*s1p != '0') *(r1p++) = *s1p;
  *r1p = '\0';

  return (r1);
}

int intmask (char *s)
{

/*  Assumes "s" is all uppercase */

  char *sptr;
  int mask;

  mask = 0;
  for (sptr = s; *sptr != '\0'; sptr++) 
    if ((*sptr >= 'A') && (*sptr <= 'Z')) mask |= 1 << (int) (*sptr - 'A');

  return (mask);
}

