/* CBMC emits ctype.h classifiers/case-mappers as bodyless FUNCTION_CALL
   externals; without the libc-body bridge ESBMC returns nondet and a valid
   classification reports FAILED where CBMC says SUCCESSFUL. */
extern int isalnum(int);
extern int isalpha(int);
extern int isblank(int);
extern int iscntrl(int);
extern int isdigit(int);
extern int isgraph(int);
extern int islower(int);
extern int isprint(int);
extern int ispunct(int);
extern int isspace(int);
extern int isupper(int);
extern int isxdigit(int);
extern int tolower(int);
extern int toupper(int);

int main(void)
{
  __CPROVER_assert(isalnum('7') != 0 && isalnum('_') == 0, "isalnum");
  __CPROVER_assert(isalpha('a') != 0 && isalpha('1') == 0, "isalpha");
  __CPROVER_assert(isblank(' ') != 0 && isblank('a') == 0, "isblank");
  __CPROVER_assert(iscntrl('\t') != 0 && iscntrl('a') == 0, "iscntrl");
  __CPROVER_assert(isdigit('5') != 0 && isdigit('x') == 0, "isdigit");
  __CPROVER_assert(isgraph('a') != 0 && isgraph(' ') == 0, "isgraph");
  __CPROVER_assert(islower('z') != 0 && islower('Z') == 0, "islower");
  __CPROVER_assert(isprint('a') != 0 && isprint('\t') == 0, "isprint");
  __CPROVER_assert(ispunct('!') != 0 && ispunct('a') == 0, "ispunct");
  __CPROVER_assert(isspace(' ') != 0 && isspace('a') == 0, "isspace");
  __CPROVER_assert(isupper('A') != 0 && isupper('a') == 0, "isupper");
  __CPROVER_assert(isxdigit('f') != 0 && isxdigit('g') == 0, "isxdigit");
  __CPROVER_assert(tolower('A') == 'a' && tolower('z') == 'z', "tolower");
  __CPROVER_assert(toupper('a') == 'A' && toupper('Z') == 'Z', "toupper");

  int c;
  __CPROVER_assume(c >= 'a' && c <= 'z');
  __CPROVER_assert(tolower(toupper(c)) == c, "symbolic round-trip");
  return 0;
}
