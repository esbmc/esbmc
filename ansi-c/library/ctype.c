
inline int isalnum(int c)
{ return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9'); }

inline int isalpha(int c)
{ return (c>='a' && c<='z') || (c>='A' && c<='Z'); }

inline int isblank(int c)
{ return c==' ' || c=='\t'; }

inline int iscntrl(int c)
{ return (c>=0 && c<='\037') || c=='\177'; }

inline int isdigit(int c)
{ return c>='0' && c<='9'; }

inline int isgraph(int c)
{ return c>='!' && c<='~'; }

inline int islower(int c)
{ return c>='a' && c<='z'; }

inline int isprint(int c)
{ return c>=' ' && c<='~'; }

inline int ispunct(int c)
{ return c=='!' ||
         c=='"' ||
         c=='#' ||
         c=='$' ||
         c=='%' ||
         c=='&' ||
         c=='\'' ||
         c=='(' ||
         c==')' ||
         c=='*' ||
         c=='+' ||
         c==',' ||
         c=='-' ||
         c=='.' ||
         c=='/' ||
         c==':' ||
         c==';' ||
         c=='<' ||
         c=='=' ||
         c=='>' ||
         c=='?' ||
         c=='@' ||
         c=='[' ||
         c=='\\' ||
         c==']' ||
         c=='^' ||
         c=='_' ||
         c=='`' ||
         c=='{' ||
         c=='|' ||
         c=='}' ||
         c=='~'; }

inline int isspace(int c)
{ return c=='\t' ||
         c=='\n' ||
         c=='\v' ||
         c=='\f' ||
         c=='\r' ||
         c==' '; }

inline int isupper(int c)
{ return c>='A' && c<='Z'; }

inline int isxdigit(int c)
{ return (c>='A' && c<='F') || (c>='a' && c<='f') || (c>='0' && c<='9'); }

inline int tolower(int c)
{ return (c>='A' && c<='Z')?c+('a'-'A'):c; }

inline int toupper(int c)
{ return (c>='a' && c<='z')?c-('a'-'A'):c; }

