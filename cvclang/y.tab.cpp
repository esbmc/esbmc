/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse yycvcparse
#define yylex   yycvclex
#define yyerror yycvcerror
#define yylval  yycvclval
#define yychar  yycvcchar
#define yydebug yycvcdebug
#define yynerrs yycvcnerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     ASSERT = 258,
     NEWLINE = 259,
     STOP = 260,
     SKIP = 261,
     CHAOS = 262,
     NAME = 263,
     NUMBER = 264,
     CSPTRUE = 265,
     CSPFALSE = 266,
     CHANNEL = 267,
     DATATYPE = 268,
     NAMETYPE = 269,
     OPEN = 270,
     CLOSE = 271,
     LSUBST = 272,
     RSUBST = 273,
     LBRACE = 274,
     RBRACE = 275,
     LPBRACE = 276,
     RPBRACE = 277,
     EQUAL = 278,
     COMMA = 279,
     DOTDOT = 280,
     PIPE = 281,
     BECOMES = 282,
     UNION = 283,
     DIFF = 284,
     INTER = 285,
     HEAD = 286,
     TAIL = 287,
     ELSE = 288,
     THEN = 289,
     IF = 290,
     LDOT = 291,
     LAMBDA = 292,
     IN = 293,
     LET = 294,
     COLON = 295,
     AT = 296,
     RSQUARE = 297,
     PAR = 298,
     LSQUARE = 299,
     RCOMM = 300,
     LCOMM = 301,
     BACKSLASH = 302,
     INTL = 303,
     NDET = 304,
     BOX = 305,
     INTR = 306,
     SEMI = 307,
     GUARD = 308,
     WITHIN = 309,
     DOT = 310,
     OR = 311,
     AND = 312,
     NOT = 313,
     EQ = 314,
     NE = 315,
     LT = 316,
     LE = 317,
     GT = 318,
     GE = 319,
     MINUS = 320,
     PLUS = 321,
     MOD = 322,
     SLASH = 323,
     TIMES = 324,
     HASH = 325,
     CAT = 326,
     ARROW = 327,
     PLING = 328,
     QUERY = 329
   };
#endif
/* Tokens.  */
#define ASSERT 258
#define NEWLINE 259
#define STOP 260
#define SKIP 261
#define CHAOS 262
#define NAME 263
#define NUMBER 264
#define CSPTRUE 265
#define CSPFALSE 266
#define CHANNEL 267
#define DATATYPE 268
#define NAMETYPE 269
#define OPEN 270
#define CLOSE 271
#define LSUBST 272
#define RSUBST 273
#define LBRACE 274
#define RBRACE 275
#define LPBRACE 276
#define RPBRACE 277
#define EQUAL 278
#define COMMA 279
#define DOTDOT 280
#define PIPE 281
#define BECOMES 282
#define UNION 283
#define DIFF 284
#define INTER 285
#define HEAD 286
#define TAIL 287
#define ELSE 288
#define THEN 289
#define IF 290
#define LDOT 291
#define LAMBDA 292
#define IN 293
#define LET 294
#define COLON 295
#define AT 296
#define RSQUARE 297
#define PAR 298
#define LSQUARE 299
#define RCOMM 300
#define LCOMM 301
#define BACKSLASH 302
#define INTL 303
#define NDET 304
#define BOX 305
#define INTR 306
#define SEMI 307
#define GUARD 308
#define WITHIN 309
#define DOT 310
#define OR 311
#define AND 312
#define NOT 313
#define EQ 314
#define NE 315
#define LT 316
#define LE 317
#define GT 318
#define GE 319
#define MINUS 320
#define PLUS 321
#define MOD 322
#define SLASH 323
#define TIMES 324
#define HASH 325
#define CAT 326
#define ARROW 327
#define PLING 328
#define QUERY 329




/* Copy the first part of user declarations.  */
#line 1 "parser.y"


#include <stdio.h>
#include <string.h>

#include <expr.h>

#include "cvc_parser.h"

#define YYMAXDEPTH 20000
#define YYINITDEPTH 20000
#define YYSTYPE unsigned

#define mto(x, y) stack(x).move_to_operands(stack(y))

extern char *yycvctext;

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(exprt &expr)
{
  expr.clear();
  PARSER.set_location(expr);
}

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(YYSTYPE &expr)
 {
  newstack(expr);
  init(stack(expr));
 }

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(YYSTYPE &expr, const std::string &id)
 {
  init(expr);
  stack(expr).id(id);
 }

/*******************************************************************\

Function: binary_op

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void binary_op(YYSTYPE &dest,
                      YYSTYPE op1,
                      const std::string &id,
                      YYSTYPE op2)
 {
  init(dest, id);
  stack(dest).move_to_operands(stack(op1), stack(op2));
 }

/*******************************************************************\

Function: yycvcerror

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int yycvcerror(char *error)
{
  PARSER.parse_error(error, yycvctext);
  return strlen(error)+1;
}

int yylex();



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 376 "y.tab.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  13
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   864

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  75
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  40
/* YYNRULES -- Number of rules.  */
#define YYNRULES  150
/* YYNRULES -- Number of states.  */
#define YYNSTATES  287

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   329

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     6,     9,    13,    17,    20,    25,
      30,    31,    33,    35,    40,    42,    44,    46,    50,    52,
      54,    56,    60,    62,    64,    68,    73,    77,    81,    85,
      89,    93,    97,   104,   112,   119,   121,   123,   128,   133,
     137,   141,   148,   155,   162,   169,   179,   181,   185,   187,
     189,   191,   193,   195,   197,   199,   201,   203,   205,   208,
     212,   215,   218,   221,   225,   229,   233,   235,   237,   239,
     241,   243,   245,   247,   249,   251,   253,   255,   257,   261,
     263,   265,   267,   270,   274,   278,   282,   286,   290,   294,
     298,   302,   304,   306,   308,   311,   314,   318,   322,   326,
     330,   334,   336,   338,   340,   342,   344,   346,   348,   350,
     352,   354,   356,   360,   366,   370,   375,   380,   387,   389,
     393,   395,   397,   399,   401,   405,   406,   408,   410,   414,
     418,   423,   428,   430,   432,   433,   435,   437,   441,   443,
     445,   449,   452,   454,   456,   458,   460,   462,   464,   466,
     468
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      76,     0,    -1,    -1,    77,    -1,     4,    76,    -1,    77,
       4,    76,    -1,    79,    23,    80,    -1,    12,   101,    -1,
      12,   101,    40,    81,    -1,    13,   100,    23,    83,    -1,
      -1,     4,    -1,   100,    -1,   100,    15,    86,    16,    -1,
      84,    -1,    88,    -1,    82,    -1,    82,    55,    81,    -1,
     100,    -1,   103,    -1,   100,    -1,   100,    26,    83,    -1,
     100,    -1,    85,    -1,    15,    85,    16,    -1,   100,    15,
      86,    16,    -1,   100,    72,    84,    -1,    89,    72,    84,
      -1,    84,    52,    84,    -1,    84,    50,    84,    -1,    84,
      49,    84,    -1,    84,    48,    84,    -1,    84,    46,   105,
      78,    45,    84,    -1,    84,    44,   105,    43,   105,    42,
      84,    -1,    35,    94,    34,    84,    33,    84,    -1,     5,
      -1,     6,    -1,     7,    15,   103,    16,    -1,     7,    15,
     100,    16,    -1,    84,    47,   105,    -1,   114,    53,    84,
      -1,    49,    87,    40,   105,    41,    84,    -1,    50,    87,
      40,   105,    41,    84,    -1,    48,    87,    40,   105,    41,
      84,    -1,    52,    87,    40,   105,    41,    84,    -1,    43,
      87,    40,   105,    41,    44,    87,    42,    84,    -1,    87,
      -1,    86,    24,    87,    -1,   100,    -1,    88,    -1,   108,
      -1,    95,    -1,    97,    -1,   114,    -1,   102,    -1,   103,
      -1,   106,    -1,    89,    -1,   100,    90,    -1,   100,    40,
      91,    -1,    55,    92,    -1,    73,    92,    -1,    74,    92,
      -1,    55,    92,    90,    -1,    73,    92,    90,    -1,    74,
      92,    90,    -1,   103,    -1,   106,    -1,   100,    -1,   100,
      -1,    97,    -1,    96,    -1,   103,    -1,   106,    -1,   108,
      -1,   102,    -1,   114,    -1,    89,    -1,    89,    24,    89,
      -1,    95,    -1,   114,    -1,    96,    -1,    58,    87,    -1,
      87,    57,    87,    -1,    87,    56,    87,    -1,    87,    59,
      87,    -1,    87,    60,    87,    -1,    87,    61,    87,    -1,
      87,    63,    87,    -1,    87,    62,    87,    -1,    87,    64,
      87,    -1,    10,    -1,    11,    -1,     9,    -1,    65,    98,
      -1,    70,    99,    -1,    98,    66,    98,    -1,    98,    65,
      98,    -1,    98,    69,    98,    -1,    98,    68,    98,    -1,
      98,    67,    98,    -1,    97,    -1,   100,    -1,   114,    -1,
     108,    -1,   103,    -1,   102,    -1,   106,    -1,   100,    -1,
     114,    -1,     8,    -1,   100,    -1,   100,    24,   101,    -1,
      15,    87,    24,    86,    16,    -1,    19,   110,    20,    -1,
      28,    15,   104,    16,    -1,    30,    15,   104,    16,    -1,
      29,    15,   105,    24,   105,    16,    -1,   105,    -1,   105,
      24,   104,    -1,   103,    -1,   106,    -1,   100,    -1,   114,
      -1,    21,   107,    22,    -1,    -1,   101,    -1,    93,    -1,
      61,   110,    63,    -1,   109,    71,   109,    -1,    31,    15,
     109,    16,    -1,    32,    15,   109,    16,    -1,   108,    -1,
     100,    -1,    -1,   111,    -1,   112,    -1,   111,    24,   112,
      -1,    96,    -1,    97,    -1,   113,    25,   113,    -1,   113,
      25,    -1,   100,    -1,   103,    -1,   108,    -1,   102,    -1,
      89,    -1,   114,    -1,    97,    -1,   100,    -1,    15,    87,
      16,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   161,   161,   162,   163,   164,   167,   172,   176,   180,
     186,   187,   190,   191,   194,   195,   198,   199,   202,   203,
     206,   207,   210,   211,   214,   215,   216,   217,   218,   219,
     220,   221,   222,   223,   224,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   242,   243,   246,   247,
     250,   251,   252,   253,   254,   255,   256,   257,   260,   261,
     264,   265,   266,   267,   268,   269,   272,   273,   274,   277,
     278,   279,   280,   281,   282,   283,   284,   287,   288,   291,
     292,   295,   296,   297,   298,   299,   300,   301,   302,   303,
     304,   307,   308,   311,   317,   318,   319,   320,   321,   322,
     323,   326,   327,   328,   331,   332,   333,   334,   335,   336,
     339,   342,   343,   346,   349,   350,   351,   352,   355,   356,
     359,   360,   361,   362,   365,   368,   369,   370,   373,   374,
     375,   376,   379,   380,   383,   384,   387,   388,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   403,   404,
     407
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "ASSERT", "NEWLINE", "STOP", "SKIP",
  "CHAOS", "NAME", "NUMBER", "CSPTRUE", "CSPFALSE", "CHANNEL", "DATATYPE",
  "NAMETYPE", "OPEN", "CLOSE", "LSUBST", "RSUBST", "LBRACE", "RBRACE",
  "LPBRACE", "RPBRACE", "EQUAL", "COMMA", "DOTDOT", "PIPE", "BECOMES",
  "UNION", "DIFF", "INTER", "HEAD", "TAIL", "ELSE", "THEN", "IF", "LDOT",
  "LAMBDA", "IN", "LET", "COLON", "AT", "RSQUARE", "PAR", "LSQUARE",
  "RCOMM", "LCOMM", "BACKSLASH", "INTL", "NDET", "BOX", "INTR", "SEMI",
  "GUARD", "WITHIN", "DOT", "OR", "AND", "NOT", "EQ", "NE", "LT", "LE",
  "GT", "GE", "MINUS", "PLUS", "MOD", "SLASH", "TIMES", "HASH", "CAT",
  "ARROW", "PLING", "QUERY", "$accept", "defns", "defn", "newline0",
  "lside", "rside", "seq_type", "type", "datatype", "proc",
  "proc_minus_name", "exprs", "expr", "expr_minus_name", "dotted",
  "dot_seq", "event_set", "d_expr", "dotteds", "cond", "bool", "b_minus",
  "num", "num_plus", "hash_arg", "name", "names", "tuple", "set",
  "setnames", "setname", "commset", "comm0", "seq", "seqname", "targ0",
  "targs", "targ", "num_name", "amb", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    75,    76,    76,    76,    76,    77,    77,    77,    77,
      78,    78,    79,    79,    80,    80,    81,    81,    82,    82,
      83,    83,    84,    84,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    86,    86,    87,    87,
      88,    88,    88,    88,    88,    88,    88,    88,    89,    89,
      90,    90,    90,    90,    90,    90,    91,    91,    91,    92,
      92,    92,    92,    92,    92,    92,    92,    93,    93,    94,
      94,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    96,    96,    97,    97,    97,    97,    97,    97,    97,
      97,    98,    98,    98,    99,    99,    99,    99,    99,    99,
     100,   101,   101,   102,   103,   103,   103,   103,   104,   104,
     105,   105,   105,   105,   106,   107,   107,   107,   108,   108,
     108,   108,   109,   109,   110,   110,   111,   111,   112,   112,
     112,   112,   112,   112,   112,   112,   112,   112,   113,   113,
     114
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     1,     2,     3,     3,     2,     4,     4,
       0,     1,     1,     4,     1,     1,     1,     3,     1,     1,
       1,     3,     1,     1,     3,     4,     3,     3,     3,     3,
       3,     3,     6,     7,     6,     1,     1,     4,     4,     3,
       3,     6,     6,     6,     6,     9,     1,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     3,
       2,     2,     2,     3,     3,     3,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     1,
       1,     1,     2,     3,     3,     3,     3,     3,     3,     3,
       3,     1,     1,     1,     2,     2,     3,     3,     3,     3,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     5,     3,     4,     4,     6,     1,     3,
       1,     1,     1,     1,     3,     0,     1,     1,     3,     3,
       4,     4,     1,     1,     0,     1,     1,     3,     1,     1,
       3,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     2,   110,     0,     0,     0,     3,     0,    12,     4,
     111,     7,     0,     1,     2,     0,     0,     0,     0,     0,
       5,    35,    36,     0,    93,    91,    92,     0,   134,   125,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   134,     0,     0,     6,    14,    23,     0,    49,
      57,    51,    81,    52,     0,    48,    54,    55,    56,    50,
       0,    53,     0,     0,    46,    49,    57,    48,    53,   112,
       8,    16,    18,    19,     9,    20,     0,     0,    23,     0,
     146,   138,   101,   102,   145,   143,   144,     0,   135,   136,
       0,   103,    77,   127,   111,   126,     0,     0,     0,     0,
       0,     0,     0,    51,    53,     0,     0,     0,     0,     0,
      82,     0,     0,   101,    94,   102,   103,    95,   108,   106,
     105,   107,   104,   109,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    58,     0,     0,    13,     0,     0,     0,     0,     0,
      24,   150,     0,   114,     0,   141,     0,   124,   122,   120,
       0,   118,   121,   123,     0,     0,   133,   132,     0,     0,
       0,     0,     0,     0,     0,     0,   128,     0,     0,    10,
      39,     0,    31,     0,    22,     0,    30,    29,    28,    84,
      83,    85,    86,    87,    89,    88,    90,    27,    97,    96,
     100,    99,    98,     0,    59,    68,    66,    67,    60,    71,
      70,    69,    75,    72,    73,    74,    76,    26,    61,    62,
     129,    40,    47,    17,    21,    38,    37,     0,   137,   101,
     102,   140,    78,     0,   115,     0,     0,   116,   130,   131,
       0,     0,     0,     0,     0,     0,     0,    11,     0,    25,
      63,    64,    65,   113,   119,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   117,    34,     0,    43,    41,    42,
      44,     0,    32,     0,    33,     0,    45
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     5,     6,   258,     7,    45,    70,    71,    74,    77,
      47,    63,    64,    65,    66,   151,   214,   218,    93,   102,
      51,    52,    53,    54,   117,    67,    11,    56,    57,   170,
     171,    58,    96,    59,    60,    87,    88,    89,    90,    68
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -124
static const yytype_int16 yypact[] =
{
     147,   147,  -124,    -6,    -6,    62,    91,    79,    54,  -124,
      86,    78,    96,  -124,   147,   601,   646,    -6,   342,    -6,
    -124,  -124,  -124,   108,  -124,  -124,  -124,   601,   714,    -6,
     122,   130,   143,   152,   153,   646,   646,   646,   646,   646,
     646,   646,   714,    15,    17,  -124,   593,  -124,   480,    31,
      99,  -124,  -124,   158,   785,   530,  -124,  -124,  -124,   103,
     104,   346,   646,    36,   480,  -124,  -124,   721,   790,  -124,
    -124,   131,  -124,  -124,  -124,   172,   342,   593,   199,   176,
    -124,  -124,    30,   694,  -124,  -124,   103,   179,   210,  -124,
     216,    46,   218,  -124,    13,  -124,   225,   496,   496,   496,
       2,     2,   220,   223,   183,   691,   740,   751,   765,   776,
     710,   195,   646,  -124,   127,  -124,  -124,  -124,   192,  -124,
    -124,  -124,   103,  -124,   496,   496,   496,   470,   470,   470,
     470,   646,   646,   646,   646,   646,   646,   646,   646,   470,
      15,    15,    15,    15,    15,   646,   375,   671,   470,   671,
     671,  -124,     2,   470,  -124,   646,   342,    -6,   248,   249,
    -124,  -124,   646,  -124,   714,    15,    -6,  -124,  -124,  -124,
     256,   250,  -124,  -124,   251,   262,  -124,  -124,   -10,    -7,
     470,   496,   496,   496,   496,   496,  -124,   260,   237,   275,
    -124,   601,    84,    99,     1,   231,   203,   242,   242,   634,
     710,   800,   236,   139,   -52,   243,  -124,  -124,   127,   127,
    -124,  -124,  -124,    68,  -124,  -124,  -124,  -124,   -15,  -124,
     158,   422,  -124,  -124,  -124,   103,   790,  -124,   -15,   -15,
    -124,  -124,   480,  -124,  -124,  -124,  -124,    72,  -124,    57,
      59,  -124,  -124,   222,  -124,   496,   496,  -124,  -124,  -124,
     413,   267,   272,   273,   284,   292,   496,  -124,   293,  -124,
    -124,  -124,  -124,  -124,  -124,   323,   470,   297,   470,   470,
     470,   470,   306,   470,  -124,   593,   646,   593,   593,   593,
     593,   470,   797,   527,   797,   470,   593
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -124,    12,  -124,  -124,  -124,  -124,   204,  -124,   194,   229,
     -22,  -123,    76,   340,   201,   -97,  -124,   -60,  -124,  -124,
     331,    29,   408,   -36,  -124,     0,    65,    23,   382,   -91,
      37,   238,  -124,   290,   -80,   327,  -124,   209,   211,   162
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -150
static const yytype_int16 yytable[] =
{
       8,     8,     2,    10,    12,    78,   248,   114,   175,   249,
       2,   137,   138,     9,     8,    55,   145,    10,    72,    75,
     178,   179,   213,     2,    24,     2,    20,    55,    83,    94,
     112,   -15,    62,    33,    34,   -15,    28,    17,    29,   237,
     147,   146,    83,   115,   118,    30,    31,    32,    33,    34,
    -139,    84,   154,   146,  -139,  -148,   147,    81,   149,   150,
     155,   152,    13,    42,   152,    84,  -147,   119,   147,    16,
    -147,    81,   230,   148,   149,   150,   158,  -148,    42,  -149,
      43,  -148,    69,  -149,   259,    44,   149,   150,   263,   228,
     229,    48,   155,  -139,    95,    14,   155,   168,   168,   168,
     176,   176,    15,    79,   208,   209,   210,   211,   212,  -147,
      17,    48,   105,   106,   107,   108,   109,   110,    18,    19,
    -148,   260,  -149,    76,   168,   168,   168,   194,   194,   194,
     194,   261,   262,   128,   129,   174,   130,    97,    79,   194,
     115,   115,   115,   115,   115,    98,   215,   221,   194,   221,
     221,     1,   176,   194,   264,     2,    72,    75,    99,     3,
       4,   188,   189,   190,    83,   240,   243,   100,   101,    78,
     222,   139,   222,   222,  -132,   152,   219,    61,   219,   219,
     194,   168,   168,   168,   168,   168,   156,    84,   187,    61,
      91,    55,   161,    81,   142,   143,   144,   104,   157,   163,
     162,   136,   137,   138,    91,   116,   123,   199,   200,   201,
     202,   203,   204,   205,   206,   160,    50,   -80,   251,   252,
     253,   254,   255,  -101,  -101,  -101,  -101,  -101,    50,    80,
      92,   232,   131,   132,   164,   133,   134,   135,   136,   137,
     138,   165,   166,    80,    46,   168,   168,   167,  -103,  -103,
    -103,  -103,  -103,   129,   180,   130,   168,   -79,   186,   173,
     173,   173,   146,  -133,   235,   236,   194,   187,   194,   194,
     194,   194,   244,   194,   245,   246,   161,   147,   247,   257,
     256,   194,   121,   265,   153,   194,   173,   173,   173,   195,
     195,   195,   195,   272,   130,   149,   150,   135,   136,   137,
     138,   195,   116,   116,   116,   116,   116,   138,   267,   226,
     195,   226,   226,   268,   269,   195,   131,   132,    86,   133,
     134,   135,   136,   137,   138,   270,    91,   116,   193,   193,
     193,   193,    86,   271,   122,   172,   172,   172,   273,   274,
     193,   276,   195,   173,   173,   173,   173,   173,   281,   193,
       2,   234,   283,    61,   193,    49,   192,   196,   197,   198,
     233,    28,   172,   172,   172,    80,   103,   242,   207,   111,
      30,    31,    32,   238,     0,     0,   241,   227,     0,     0,
       0,   193,   231,     2,   217,   224,     0,   224,   224,     0,
     177,   177,    50,     0,    28,     0,    29,     0,     0,   153,
      73,     0,     0,    30,    31,    32,     0,   173,   173,   250,
      85,  -103,  -103,  -103,  -103,  -103,     0,     0,   173,   172,
     172,   172,   172,   172,    85,     0,   120,     0,   195,     0,
     195,   195,   195,   195,     0,   195,    82,   225,     0,   225,
     225,     0,   177,   195,     0,     0,   266,   195,     0,     0,
      82,   113,     0,     0,    86,     0,     0,   124,   159,   125,
     126,   127,   128,   129,     0,   130,     0,   193,     0,   193,
     193,   193,   193,     0,   193,    21,    22,    23,     2,   169,
     169,   169,   193,   172,   172,   191,   193,  -102,  -102,  -102,
    -102,  -102,     0,  -133,   172,   275,     0,   277,   278,   279,
     280,     0,   282,     0,     2,    35,   169,   169,   169,     0,
     284,   112,     0,    36,   286,    28,     0,    29,    37,    38,
      39,     0,    40,     0,    30,    31,    32,     0,   216,   223,
     -22,   223,   223,     0,   -22,     0,   131,   132,    73,   133,
     134,   135,   136,   137,   138,   145,    85,     0,   113,   113,
     113,   113,   113,     0,     0,   220,     0,   220,   220,     0,
       0,     0,     0,   169,   169,   169,   169,   169,     0,   285,
     146,     0,    82,   239,   -22,     0,   -22,   -22,   -22,   -22,
     -22,     0,   -22,   131,   132,   147,   133,   134,   135,   136,
     137,   138,     0,     0,     0,  -102,  -102,  -102,  -102,  -102,
       0,  -133,   148,   149,   150,     0,    21,    22,    23,     2,
      24,    25,    26,     0,     0,     0,    27,     0,     0,     0,
      28,     0,    29,     0,     0,     0,     0,   169,   169,    30,
      31,    32,    33,    34,     0,     0,    35,   124,   169,   125,
     126,   127,   128,   129,    36,   130,     0,     0,     0,    37,
      38,    39,     0,    40,     2,    24,    25,    26,     0,    41,
       0,    62,    42,     0,     0,    28,    43,    29,     0,     0,
       0,    44,     0,     0,    30,    31,    32,    33,    34,     2,
      24,    25,    26,     0,     0,     0,    62,     0,     0,     0,
      28,   132,    29,   133,   134,   135,   136,   137,   138,    30,
      31,    32,    33,    34,    41,     0,     0,    42,     0,     0,
       0,    43,     0,     0,  -142,     0,    44,     0,  -142,  -149,
       0,     0,     2,    24,    25,    26,     0,     0,     0,    62,
       0,   181,    42,    28,   146,     0,    43,     0,     0,     0,
       0,    44,    30,    31,    32,    33,    34,   131,   132,   147,
     133,   134,   135,   136,   137,   138,     0,  -142,     0,     0,
       0,   146,     0,     0,     0,  -133,     0,   149,   150,   133,
     134,   135,   136,   137,   138,    42,   147,     0,     0,    43,
     182,     0,     0,     0,    44,     0,  -102,  -102,  -102,  -102,
    -102,   183,  -133,     0,   149,   150,   131,   132,     0,   133,
     134,   135,   136,   137,   138,   184,     0,   131,   132,     0,
     133,   134,   135,   136,   137,   138,   185,     0,     0,     0,
       0,   131,   132,     0,   133,   134,   135,   136,   137,   138,
       0,     0,   131,   132,     0,   133,   134,   135,   136,   137,
     138,  -150,     0,  -150,   126,   127,   128,   129,     0,   130,
     140,   141,   142,   143,   144,  -103,  -103,  -103,  -103,  -103,
     134,   135,   136,   137,   138
};

static const yytype_int16 yycheck[] =
{
       0,     1,     8,     3,     4,    27,    16,    43,    99,    16,
       8,    63,    64,     1,    14,    15,    15,    17,    18,    19,
     100,   101,   145,     8,     9,     8,    14,    27,    28,    29,
      15,     0,    15,    31,    32,     4,    19,    24,    21,   162,
      55,    40,    42,    43,    44,    28,    29,    30,    31,    32,
      20,    28,    16,    40,    24,    25,    55,    28,    73,    74,
      24,    71,     0,    61,    71,    42,    20,    44,    55,    15,
      24,    42,   152,    72,    73,    74,    76,    20,    61,    20,
      65,    24,    17,    24,    16,    70,    73,    74,    16,   149,
     150,    15,    24,    63,    29,     4,    24,    97,    98,    99,
     100,   101,    23,    27,   140,   141,   142,   143,   144,    63,
      24,    35,    36,    37,    38,    39,    40,    41,    40,    23,
      63,   218,    63,    15,   124,   125,   126,   127,   128,   129,
     130,   228,   229,    49,    50,    98,    52,    15,    62,   139,
     140,   141,   142,   143,   144,    15,   146,   147,   148,   149,
     150,     4,   152,   153,   245,     8,   156,   157,    15,    12,
      13,   124,   125,   126,   164,   165,   166,    15,    15,   191,
     147,    72,   149,   150,    71,    71,   147,    15,   149,   150,
     180,   181,   182,   183,   184,   185,    55,   164,   112,    27,
      28,   191,    16,   164,    67,    68,    69,    35,    26,    20,
      24,    62,    63,    64,    42,    43,    44,   131,   132,   133,
     134,   135,   136,   137,   138,    16,    15,    34,   181,   182,
     183,   184,   185,    65,    66,    67,    68,    69,    27,    28,
      29,   155,    56,    57,    24,    59,    60,    61,    62,    63,
      64,    25,    24,    42,    15,   245,   246,    22,    65,    66,
      67,    68,    69,    50,    34,    52,   256,    34,    63,    97,
      98,    99,    40,    71,    16,    16,   266,   191,   268,   269,
     270,   271,    16,   273,    24,    24,    16,    55,    16,     4,
      43,   281,    44,   246,    53,   285,   124,   125,   126,   127,
     128,   129,   130,   256,    52,    73,    74,    61,    62,    63,
      64,   139,   140,   141,   142,   143,   144,    64,    41,   147,
     148,   149,   150,    41,    41,   153,    56,    57,    28,    59,
      60,    61,    62,    63,    64,    41,   164,   165,   127,   128,
     129,   130,    42,    41,    44,    97,    98,    99,    45,    16,
     139,    44,   180,   181,   182,   183,   184,   185,    42,   148,
       8,   157,   276,   191,   153,    15,   127,   128,   129,   130,
     156,    19,   124,   125,   126,   164,    35,   166,   139,    42,
      28,    29,    30,   164,    -1,    -1,   165,   148,    -1,    -1,
      -1,   180,   153,     8,   146,   147,    -1,   149,   150,    -1,
     100,   101,   191,    -1,    19,    -1,    21,    -1,    -1,    53,
      18,    -1,    -1,    28,    29,    30,    -1,   245,   246,   180,
      28,    65,    66,    67,    68,    69,    -1,    -1,   256,   181,
     182,   183,   184,   185,    42,    -1,    44,    -1,   266,    -1,
     268,   269,   270,   271,    -1,   273,    28,   147,    -1,   149,
     150,    -1,   152,   281,    -1,    -1,    33,   285,    -1,    -1,
      42,    43,    -1,    -1,   164,    -1,    -1,    44,    76,    46,
      47,    48,    49,    50,    -1,    52,    -1,   266,    -1,   268,
     269,   270,   271,    -1,   273,     5,     6,     7,     8,    97,
      98,    99,   281,   245,   246,    15,   285,    65,    66,    67,
      68,    69,    -1,    71,   256,   266,    -1,   268,   269,   270,
     271,    -1,   273,    -1,     8,    35,   124,   125,   126,    -1,
     281,    15,    -1,    43,   285,    19,    -1,    21,    48,    49,
      50,    -1,    52,    -1,    28,    29,    30,    -1,   146,   147,
       0,   149,   150,    -1,     4,    -1,    56,    57,   156,    59,
      60,    61,    62,    63,    64,    15,   164,    -1,   140,   141,
     142,   143,   144,    -1,    -1,   147,    -1,   149,   150,    -1,
      -1,    -1,    -1,   181,   182,   183,   184,   185,    -1,    42,
      40,    -1,   164,   165,    44,    -1,    46,    47,    48,    49,
      50,    -1,    52,    56,    57,    55,    59,    60,    61,    62,
      63,    64,    -1,    -1,    -1,    65,    66,    67,    68,    69,
      -1,    71,    72,    73,    74,    -1,     5,     6,     7,     8,
       9,    10,    11,    -1,    -1,    -1,    15,    -1,    -1,    -1,
      19,    -1,    21,    -1,    -1,    -1,    -1,   245,   246,    28,
      29,    30,    31,    32,    -1,    -1,    35,    44,   256,    46,
      47,    48,    49,    50,    43,    52,    -1,    -1,    -1,    48,
      49,    50,    -1,    52,     8,     9,    10,    11,    -1,    58,
      -1,    15,    61,    -1,    -1,    19,    65,    21,    -1,    -1,
      -1,    70,    -1,    -1,    28,    29,    30,    31,    32,     8,
       9,    10,    11,    -1,    -1,    -1,    15,    -1,    -1,    -1,
      19,    57,    21,    59,    60,    61,    62,    63,    64,    28,
      29,    30,    31,    32,    58,    -1,    -1,    61,    -1,    -1,
      -1,    65,    -1,    -1,    20,    -1,    70,    -1,    24,    25,
      -1,    -1,     8,     9,    10,    11,    -1,    -1,    -1,    15,
      -1,    40,    61,    19,    40,    -1,    65,    -1,    -1,    -1,
      -1,    70,    28,    29,    30,    31,    32,    56,    57,    55,
      59,    60,    61,    62,    63,    64,    -1,    63,    -1,    -1,
      -1,    40,    -1,    -1,    -1,    71,    -1,    73,    74,    59,
      60,    61,    62,    63,    64,    61,    55,    -1,    -1,    65,
      40,    -1,    -1,    -1,    70,    -1,    65,    66,    67,    68,
      69,    40,    71,    -1,    73,    74,    56,    57,    -1,    59,
      60,    61,    62,    63,    64,    40,    -1,    56,    57,    -1,
      59,    60,    61,    62,    63,    64,    40,    -1,    -1,    -1,
      -1,    56,    57,    -1,    59,    60,    61,    62,    63,    64,
      -1,    -1,    56,    57,    -1,    59,    60,    61,    62,    63,
      64,    44,    -1,    46,    47,    48,    49,    50,    -1,    52,
      65,    66,    67,    68,    69,    65,    66,    67,    68,    69,
      60,    61,    62,    63,    64
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     4,     8,    12,    13,    76,    77,    79,   100,    76,
     100,   101,   100,     0,     4,    23,    15,    24,    40,    23,
      76,     5,     6,     7,     9,    10,    11,    15,    19,    21,
      28,    29,    30,    31,    32,    35,    43,    48,    49,    50,
      52,    58,    61,    65,    70,    80,    84,    85,    87,    88,
      89,    95,    96,    97,    98,   100,   102,   103,   106,   108,
     109,   114,    15,    86,    87,    88,    89,   100,   114,   101,
      81,    82,   100,   103,    83,   100,    15,    84,    85,    87,
      89,    96,    97,   100,   102,   103,   108,   110,   111,   112,
     113,   114,    89,    93,   100,   101,   107,    15,    15,    15,
      15,    15,    94,    95,   114,    87,    87,    87,    87,    87,
      87,   110,    15,    97,    98,   100,   114,    99,   100,   102,
     103,   106,   108,   114,    44,    46,    47,    48,    49,    50,
      52,    56,    57,    59,    60,    61,    62,    63,    64,    72,
      65,    66,    67,    68,    69,    15,    40,    55,    72,    73,
      74,    90,    71,    53,    16,    24,    55,    26,   100,   103,
      16,    16,    24,    20,    24,    25,    24,    22,   100,   103,
     104,   105,   106,   114,   105,   104,   100,   108,   109,   109,
      34,    40,    40,    40,    40,    40,    63,    87,   105,   105,
     105,    15,    84,    89,   100,   114,    84,    84,    84,    87,
      87,    87,    87,    87,    87,    87,    87,    84,    98,    98,
      98,    98,    98,    86,    91,   100,   103,   106,    92,    96,
      97,   100,   102,   103,   106,   108,   114,    84,    92,    92,
     109,    84,    87,    81,    83,    16,    16,    86,   112,    97,
     100,   113,    89,   100,    16,    24,    24,    16,    16,    16,
      84,   105,   105,   105,   105,   105,    43,     4,    78,    16,
      90,    90,    90,    16,   104,   105,    33,    41,    41,    41,
      41,    41,   105,    45,    16,    84,    44,    84,    84,    84,
      84,    42,    84,    87,    84,    42,    84
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:
#line 168 "parser.y"
    {
	    init((yyval), "=");
            stack((yyval)).move_to_operands(stack((yyvsp[(1) - (3)])), stack((yyvsp[(3) - (3)])));
	   }
    break;

  case 7:
#line 173 "parser.y"
    {
            init((yyval), "channel");
           }
    break;

  case 8:
#line 177 "parser.y"
    {
            init((yyval), "channel");
           }
    break;

  case 9:
#line 181 "parser.y"
    {
            init((yyval), "datatype"); 
           }
    break;

  case 13:
#line 191 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 17:
#line 199 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 21:
#line 207 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 24:
#line 214 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); }
    break;

  case 25:
#line 215 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 26:
#line 216 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "prefixed", (yyvsp[(3) - (3)])); }
    break;

  case 27:
#line 217 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 28:
#line 218 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "sequential", (yyvsp[(3) - (3)])); }
    break;

  case 29:
#line 219 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "external_choice", (yyvsp[(3) - (3)])); }
    break;

  case 30:
#line 220 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "internal_choice", (yyvsp[(3) - (3)])); }
    break;

  case 31:
#line 221 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 32:
#line 222 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 33:
#line 223 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 34:
#line 225 "parser.y"
    {
            init((yyval), "ifthenelse");
            stack((yyval)).move_to_operands(stack((yyvsp[(2) - (6)])), stack((yyvsp[(4) - (6)])), stack((yyvsp[(6) - (6)])));
           }
    break;

  case 35:
#line 229 "parser.y"
    { init((yyval), "STOP"); }
    break;

  case 36:
#line 230 "parser.y"
    { init((yyval), "SKIP"); }
    break;

  case 37:
#line 231 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 38:
#line 232 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 39:
#line 233 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "hide", (yyvsp[(3) - (3)])); }
    break;

  case 40:
#line 234 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 41:
#line 235 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 42:
#line 236 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 43:
#line 237 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 44:
#line 238 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 45:
#line 239 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 46:
#line 242 "parser.y"
    { init((yyval)); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 47:
#line 243 "parser.y"
    { mto((yyvsp[(1) - (3)]), (yyvsp[(3) - (3)])); }
    break;

  case 58:
#line 260 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 59:
#line 261 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 60:
#line 264 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 61:
#line 265 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 62:
#line 266 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 63:
#line 267 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 64:
#line 268 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 65:
#line 269 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 77:
#line 287 "parser.y"
    { init((yyval)); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 78:
#line 288 "parser.y"
    { mto((yyval), (yyvsp[(1) - (3)])); }
    break;

  case 82:
#line 296 "parser.y"
    { init((yyval), "not"); mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 83:
#line 297 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "and", (yyvsp[(3) - (3)])); }
    break;

  case 84:
#line 298 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "or",  (yyvsp[(3) - (3)]));  }
    break;

  case 85:
#line 299 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "=",   (yyvsp[(3) - (3)]));   }
    break;

  case 86:
#line 300 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "notequal", (yyvsp[(3) - (3)])); }
    break;

  case 87:
#line 301 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "<",   (yyvsp[(3) - (3)]));   }
    break;

  case 88:
#line 302 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), ">",   (yyvsp[(3) - (3)]));   }
    break;

  case 89:
#line 303 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "<=",  (yyvsp[(3) - (3)]));  }
    break;

  case 90:
#line 304 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), ">=",  (yyvsp[(3) - (3)]));  }
    break;

  case 91:
#line 307 "parser.y"
    { init((yyval)); stack((yyval)).make_true();  }
    break;

  case 92:
#line 308 "parser.y"
    { init((yyval)); stack((yyval)).make_false(); }
    break;

  case 93:
#line 312 "parser.y"
    {
            init((yyval), "constant");
            stack((yyval)).set("value", stack((yyvsp[(1) - (1)])).id());
            stack((yyval)).type()=typet("integer");
           }
    break;

  case 94:
#line 317 "parser.y"
    { init((yyval), "unary-"); mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 95:
#line 318 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 96:
#line 319 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "+", (yyvsp[(3) - (3)])); }
    break;

  case 97:
#line 320 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "-", (yyvsp[(3) - (3)])); }
    break;

  case 98:
#line 321 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "*", (yyvsp[(3) - (3)])); }
    break;

  case 99:
#line 322 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "/", (yyvsp[(3) - (3)])); }
    break;

  case 100:
#line 323 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "%", (yyvsp[(3) - (3)])); }
    break;

  case 110:
#line 339 "parser.y"
    { init((yyval), "name"); stack((yyval)).set("identifier", stack((yyvsp[(1) - (1)])).id()); }
    break;

  case 111:
#line 342 "parser.y"
    { init((yyval)); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 112:
#line 343 "parser.y"
    { mto((yyval), (yyvsp[(1) - (3)])); }
    break;

  case 113:
#line 346 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 114:
#line 349 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 115:
#line 350 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 116:
#line 351 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 117:
#line 352 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 118:
#line 355 "parser.y"
    { init((yyval)); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 119:
#line 356 "parser.y"
    { mto((yyval), (yyvsp[(3) - (3)])); }
    break;

  case 124:
#line 365 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 128:
#line 373 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 129:
#line 374 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 130:
#line 375 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 131:
#line 376 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 134:
#line 383 "parser.y"
    { init((yyval)); }
    break;

  case 136:
#line 387 "parser.y"
    { init((yyval)); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 137:
#line 388 "parser.y"
    { mto((yyval), (yyvsp[(3) - (3)])); }
    break;

  case 140:
#line 393 "parser.y"
    { binary_op((yyval), (yyvsp[(1) - (3)]), "dotdot", (yyvsp[(3) - (3)])); }
    break;

  case 141:
#line 394 "parser.y"
    { init((yyval), "unknown"); }
    break;

  case 150:
#line 407 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); }
    break;


/* Line 1267 of yacc.c.  */
#line 2373 "y.tab.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 409 "parser.y"


