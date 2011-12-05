
/* stream.h code */


#include <stdio.h>
#define SUCCEED 1
#define FAIL 0

typedef
       FILE *character_stream;
typedef 
       int BOOLEAN;
typedef 
       char CHARACTER;
typedef
       char *string;

extern char get_char();
extern char unget_char();
extern int is_end_of_character_stream();
extern character_stream open_character_stream();

