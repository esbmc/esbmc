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
#define yyparse yybpparse
#define yylex   yybplex
#define yyerror yybperror
#define yylval  yybplval
#define yychar  yybpchar
#define yydebug yybpdebug
#define yynerrs yybpnerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     BEGIN_Token = 258,
     END_Token = 259,
     DECL_Token = 260,
     VOID_Token = 261,
     BOOL_Token = 262,
     GOTO_Token = 263,
     ENFORCE_Token = 264,
     IF_Token = 265,
     THEN_Token = 266,
     ELSE_Token = 267,
     FI_Token = 268,
     SKIP_Token = 269,
     WHILE_Token = 270,
     DO_Token = 271,
     OD_Token = 272,
     ABORTIF_Token = 273,
     START_THREAD_Token = 274,
     END_THREAD_Token = 275,
     SYNC_Token = 276,
     ATOMIC_BEGIN_Token = 277,
     ATOMIC_END_Token = 278,
     DEAD_Token = 279,
     RETURN_Token = 280,
     SCHOOSE_Token = 281,
     IDENTIFIER_Token = 282,
     NUMBER_Token = 283,
     DFS_Token = 284,
     ASSIGN_Token = 285,
     CONSTRAIN_Token = 286,
     NEQ_Token = 287,
     EQ_Token = 288,
     IMPLIES_Token = 289,
     EQUIV_Token = 290,
     XOR_Token = 291,
     OR_Token = 292,
     AND_Token = 293,
     TICK_Token = 294
   };
#endif
/* Tokens.  */
#define BEGIN_Token 258
#define END_Token 259
#define DECL_Token 260
#define VOID_Token 261
#define BOOL_Token 262
#define GOTO_Token 263
#define ENFORCE_Token 264
#define IF_Token 265
#define THEN_Token 266
#define ELSE_Token 267
#define FI_Token 268
#define SKIP_Token 269
#define WHILE_Token 270
#define DO_Token 271
#define OD_Token 272
#define ABORTIF_Token 273
#define START_THREAD_Token 274
#define END_THREAD_Token 275
#define SYNC_Token 276
#define ATOMIC_BEGIN_Token 277
#define ATOMIC_END_Token 278
#define DEAD_Token 279
#define RETURN_Token 280
#define SCHOOSE_Token 281
#define IDENTIFIER_Token 282
#define NUMBER_Token 283
#define DFS_Token 284
#define ASSIGN_Token 285
#define CONSTRAIN_Token 286
#define NEQ_Token 287
#define EQ_Token 288
#define IMPLIES_Token 289
#define EQUIV_Token 290
#define XOR_Token 291
#define OR_Token 292
#define AND_Token 293
#define TICK_Token 294




/* Copy the first part of user declarations.  */
#line 1 "parser.y"

#include <i2string.h>

#include "bp_parser.h"
#include "bp_typecheck.h"

#define YYSTYPE unsigned
#define PARSER bp_parser

#include "y.tab.h"

#define YYMAXDEPTH 200000
#define YYSTYPE_IS_TRIVIAL 1

/*------------------------------------------------------------------------*/

#define yylineno yybplineno
#define yytext yybptext

#define yyerror yybperror
int yybperror(const std::string &error);
int yylex();
extern char *yytext;

/*------------------------------------------------------------------------*/

#define mto(x, y) stack(x).move_to_operands(stack(y))
#define binary(x, y, id, z) { init(x, id); \
  stack(x).reserve_operands(2); mto(x, y); mto(x, z); }

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(exprt &expr)
{
  expr.clear();

  locationt &location=expr.location();
  location.set_line(PARSER.line_no);

  if(PARSER.filename!="")
    location.set_file(PARSER.filename);

  if(PARSER.function!="")
    location.set_function(PARSER.function);
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

Function: j_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void j_binary(YYSTYPE &dest, YYSTYPE &op1,
                     const std::string &id, YYSTYPE &op2)
{
  if(stack(op1).id()==id)
  {
    dest=op1;
    mto(dest, op2);
  }
  else if(stack(op2).id()==id)
  {
    dest=op2;
    mto(dest, op1);
  }
  else
    binary(dest, op1, id, op2);
}

/*******************************************************************\

Function: statement

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void statement(YYSTYPE &dest, const std::string &statement)
{
  init(dest, "code");
  stack(dest).set("statement", statement);
}

/*******************************************************************\

Function: new_declaration

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void new_declaration(YYSTYPE &decl)
{
  PARSER.parse_tree.declarations.push_back(exprt());
  PARSER.parse_tree.declarations.back().swap(stack(decl));
}

/*------------------------------------------------------------------------*/



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
#line 353 "y.tab.cpp"

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
#define YYFINAL  12
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   135

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  52
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  46
/* YYNRULES -- Number of rules.  */
#define YYNRULES  91
/* YYNRULES -- Number of states.  */
#define YYNSTATES  140

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   294

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    50,     2,     2,     2,     2,     2,     2,
      42,    43,    47,     2,    41,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    46,    40,
      44,     2,    45,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    48,     2,    49,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    51,     2,     2,     2,
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
      35,    36,    37,    38,    39
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] =
{
       0,     0,     3,     5,     7,    10,    13,    15,    18,    20,
      24,    26,    30,    31,    33,    34,    36,    40,    42,    44,
      49,    50,    51,    59,    60,    62,    64,    68,    69,    72,
      74,    78,    80,    86,    94,    97,    99,   103,   108,   110,
     113,   115,   118,   120,   123,   126,   128,   131,   133,   135,
     138,   140,   142,   144,   146,   148,   150,   152,   154,   156,
     158,   160,   162,   164,   166,   168,   170,   172,   173,   175,
     178,   183,   187,   191,   193,   195,   197,   199,   203,   207,
     211,   215,   219,   223,   227,   229,   231,   236,   238,   240,
     243,   246
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      53,     0,    -1,    54,    -1,    55,    -1,    54,    55,    -1,
      56,    40,    -1,    63,    -1,     5,    57,    -1,    94,    -1,
      57,    41,    94,    -1,    95,    -1,    58,    41,    95,    -1,
      -1,    57,    -1,    -1,    58,    -1,    42,    59,    43,    -1,
       6,    -1,     7,    -1,     7,    44,    28,    45,    -1,    -1,
      -1,    66,    62,    93,    64,    61,    67,    65,    -1,    -1,
      29,    -1,    68,    -1,     3,    69,     4,    -1,    -1,    70,
      40,    -1,    70,    -1,    70,    40,    86,    -1,    86,    -1,
      10,    95,    11,    69,    13,    -1,    10,    95,    11,    69,
      12,    69,    13,    -1,     8,    73,    -1,    92,    -1,    73,
      41,    92,    -1,    93,    42,    60,    43,    -1,    74,    -1,
      25,    58,    -1,    25,    -1,     9,    95,    -1,    14,    -1,
      18,    95,    -1,    19,    86,    -1,    20,    -1,    21,    27,
      -1,    22,    -1,    23,    -1,    24,    58,    -1,    77,    -1,
      79,    -1,    78,    -1,    85,    -1,    72,    -1,    75,    -1,
      76,    -1,    71,    -1,    91,    -1,    90,    -1,    87,    -1,
      80,    -1,    81,    -1,    82,    -1,    83,    -1,    84,    -1,
      56,    -1,    -1,    89,    -1,    31,    95,    -1,    58,    30,
      58,    88,    -1,    58,    30,    74,    -1,    92,    46,    86,
      -1,    27,    -1,    27,    -1,    27,    -1,    97,    -1,    95,
      34,    95,    -1,    95,    38,    95,    -1,    95,    37,    95,
      -1,    95,    36,    95,    -1,    95,    33,    95,    -1,    95,
      32,    95,    -1,    42,    95,    43,    -1,    47,    -1,    94,
      -1,    26,    48,    58,    49,    -1,    28,    -1,    96,    -1,
      50,    97,    -1,    51,    97,    -1,    39,    97,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   187,   187,   190,   191,   194,   195,   198,   204,   206,
     210,   212,   216,   217,   220,   221,   224,   227,   228,   229,
     237,   239,   236,   249,   250,   253,   256,   259,   260,   261,
     265,   266,   269,   272,   278,   293,   297,   303,   313,   316,
     320,   323,   331,   337,   343,   349,   354,   360,   365,   370,
     376,   377,   378,   379,   380,   381,   382,   383,   384,   385,
     386,   387,   388,   389,   390,   391,   394,   401,   405,   408,
     414,   432,   439,   447,   450,   456,   463,   464,   465,   466,
     467,   468,   469,   472,   473,   474,   475,   480,   488,   489,
     490,   491
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "BEGIN_Token", "END_Token", "DECL_Token",
  "VOID_Token", "BOOL_Token", "GOTO_Token", "ENFORCE_Token", "IF_Token",
  "THEN_Token", "ELSE_Token", "FI_Token", "SKIP_Token", "WHILE_Token",
  "DO_Token", "OD_Token", "ABORTIF_Token", "START_THREAD_Token",
  "END_THREAD_Token", "SYNC_Token", "ATOMIC_BEGIN_Token",
  "ATOMIC_END_Token", "DEAD_Token", "RETURN_Token", "SCHOOSE_Token",
  "IDENTIFIER_Token", "NUMBER_Token", "DFS_Token", "ASSIGN_Token",
  "CONSTRAIN_Token", "NEQ_Token", "EQ_Token", "IMPLIES_Token",
  "EQUIV_Token", "XOR_Token", "OR_Token", "AND_Token", "TICK_Token", "';'",
  "','", "'('", "')'", "'<'", "'>'", "':'", "'*'", "'['", "']'", "'!'",
  "'~'", "$accept", "start", "declarations", "declaration",
  "variable_declaration", "variable_list", "expression_list",
  "variable_list_opt", "expression_list_opt", "argument_list", "type",
  "function_definition", "@1", "@2", "dfs_opt", "function_body",
  "block_statement", "statement_list", "statement_list_rec",
  "if_statement", "goto_statement", "label_list", "function_call",
  "function_call_statement", "return_statement", "enforce_statement",
  "skip_statement", "abortif_statement", "start_thread_statement",
  "end_thread_statement", "sync_statement", "atomic_begin_statement",
  "atomic_end_statement", "dead_statement", "statement", "decl_statement",
  "constrain_opt", "constrain", "assignment_statement",
  "labeled_statement", "label", "function_name", "variable_name",
  "expression", "primary_expression", "unary_expression", 0
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
      59,    44,    40,    41,    60,    62,    58,    42,    91,    93,
      33,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    52,    53,    54,    54,    55,    55,    56,    57,    57,
      58,    58,    59,    59,    60,    60,    61,    62,    62,    62,
      64,    65,    63,    66,    66,    67,    68,    69,    69,    69,
      70,    70,    71,    71,    72,    73,    73,    74,    75,    76,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    87,    88,    88,    89,
      90,    90,    91,    92,    93,    94,    95,    95,    95,    95,
      95,    95,    95,    96,    96,    96,    96,    96,    97,    97,
      97,    97
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     2,     2,     1,     2,     1,     3,
       1,     3,     0,     1,     0,     1,     3,     1,     1,     4,
       0,     0,     7,     0,     1,     1,     3,     0,     2,     1,
       3,     1,     5,     7,     2,     1,     3,     4,     1,     2,
       1,     2,     1,     2,     2,     1,     2,     1,     1,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     0,     1,     2,
       4,     3,     3,     1,     1,     1,     1,     3,     3,     3,
       3,     3,     3,     3,     1,     1,     4,     1,     1,     2,
       2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
      23,     0,    24,     0,    23,     3,     0,     6,     0,    75,
       7,     8,     1,     4,     5,    17,    18,     0,     0,     0,
      74,    20,     9,     0,     0,    19,    12,     0,    13,     0,
      27,    21,    25,    16,     0,     0,     0,    42,     0,     0,
      45,     0,    47,    48,     0,    40,     0,    75,    87,     0,
       0,    84,     0,     0,    66,     0,     0,    29,    57,    54,
      38,    55,    56,    50,    52,    51,    61,    62,    63,    64,
      65,    53,    31,    60,    59,    58,     0,     0,    85,    10,
      88,    76,    22,    73,    34,    35,    41,     0,    43,    44,
      46,    49,    39,     0,    91,     0,    89,    90,     0,     0,
      26,    28,     0,    14,     0,     0,     0,     0,     0,     0,
       0,    27,     0,    83,    75,    67,    71,    11,    30,    72,
      15,     0,    82,    81,    77,    80,    79,    78,    36,     0,
      86,     0,    70,    68,    37,    27,    32,    69,     0,    33
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,     5,    54,    10,    55,    29,   121,    27,
      17,     7,    24,    82,     8,    31,    32,    56,    57,    58,
      59,    84,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,   132,   133,    74,    75,
      76,    77,    78,    79,    80,    81
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -98
static const yytype_int8 yypact[] =
{
       5,    14,   -98,    44,     4,   -98,     6,   -98,    69,   -98,
      36,   -98,   -98,   -98,   -98,   -98,    22,    57,    14,    62,
     -98,   -98,   -98,    46,    51,   -98,    14,    91,    36,    54,
       3,   -98,   -98,   -98,    72,    53,    53,   -98,    53,     3,
     -98,    75,   -98,   -98,    53,    53,    60,    10,   -98,    53,
      53,   -98,    53,    53,   -98,     7,   101,    67,   -98,   -98,
     -98,   -98,   -98,   -98,   -98,   -98,   -98,   -98,   -98,   -98,
     -98,   -98,   -98,   -98,   -98,   -98,    68,    84,   -98,    87,
     -98,   -98,   -98,   -98,    77,   -98,    87,    25,    87,   -98,
     -98,    86,    86,    53,   -98,    79,   -98,   -98,    59,    53,
     -98,     3,     3,    53,    53,    53,    53,    53,    53,    53,
      72,     3,    -9,   -98,    88,     8,   -98,    87,   -98,   -98,
      86,    85,    87,    87,    87,    45,    93,   -98,   -98,    76,
     -98,    53,   -98,   -98,   -98,     3,   -98,    87,   116,   -98
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -98,   -98,   -98,   128,    47,   107,   -38,   -98,   -98,   -98,
     -98,   -98,   -98,   -98,   -98,   -98,   -98,   -97,   -98,   -98,
     -98,   -98,    37,   -98,   -98,   -98,   -98,   -98,   -98,   -98,
     -98,   -98,   -98,   -98,   -34,   -98,   -98,   -98,   -98,   -98,
     -32,   117,    17,   -35,   -98,   -33
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -75
static const yytype_int16 yytable[] =
{
      86,    87,    85,    88,    -2,    89,    91,    92,     1,     1,
       1,    34,    35,    36,   129,    95,    94,    37,    11,    96,
      97,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    99,     2,     2,    22,   111,    98,   138,   131,
     130,     9,    49,    11,    12,    50,    14,     6,    99,    99,
      51,     6,   -74,    52,    53,   112,   -73,   104,   105,   106,
     115,   107,   108,   109,   117,   120,    19,   118,   119,   122,
     123,   124,   125,   126,   127,    15,    16,    18,   128,    46,
       9,    48,   108,   109,    20,    46,   114,    48,   135,   136,
      23,    25,    49,    26,    30,    50,   137,    33,    49,    83,
      51,    50,    90,    52,    53,   100,    51,   101,    93,    52,
      53,   104,   105,   106,   102,   107,   108,   109,   110,   104,
     105,   106,   113,   107,   108,   109,   103,    99,   134,   139,
     -74,   109,    13,    28,    21,   116
};

static const yytype_uint8 yycheck[] =
{
      35,    36,    34,    38,     0,    39,    44,    45,     5,     5,
       5,     8,     9,    10,   111,    50,    49,    14,     1,    52,
      53,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    41,    29,    29,    18,    11,    30,   135,    31,
      49,    27,    39,    26,     0,    42,    40,     0,    41,    41,
      47,     4,    42,    50,    51,    93,    46,    32,    33,    34,
      98,    36,    37,    38,    99,   103,    44,   101,   102,   104,
     105,   106,   107,   108,   109,     6,     7,    41,   110,    26,
      27,    28,    37,    38,    27,    26,    27,    28,    12,    13,
      28,    45,    39,    42,     3,    42,   131,    43,    39,    27,
      47,    42,    27,    50,    51,     4,    47,    40,    48,    50,
      51,    32,    33,    34,    46,    36,    37,    38,    41,    32,
      33,    34,    43,    36,    37,    38,    42,    41,    43,    13,
      42,    38,     4,    26,    17,    98
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     5,    29,    53,    54,    55,    56,    63,    66,    27,
      57,    94,     0,    55,    40,     6,     7,    62,    41,    44,
      27,    93,    94,    28,    64,    45,    42,    61,    57,    59,
       3,    67,    68,    43,     8,     9,    10,    14,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    39,
      42,    47,    50,    51,    56,    58,    69,    70,    71,    72,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    90,    91,    92,    93,    94,    95,
      96,    97,    65,    27,    73,    92,    95,    95,    95,    86,
      27,    58,    58,    48,    97,    95,    97,    97,    30,    41,
       4,    40,    46,    42,    32,    33,    34,    36,    37,    38,
      41,    11,    58,    43,    27,    58,    74,    95,    86,    86,
      58,    60,    95,    95,    95,    95,    95,    95,    92,    69,
      49,    31,    88,    89,    43,    12,    13,    95,    69,    13
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
        case 3:
#line 190 "parser.y"
    { new_declaration((yyvsp[(1) - (1)])); }
    break;

  case 4:
#line 191 "parser.y"
    { new_declaration((yyvsp[(2) - (2)])); }
    break;

  case 7:
#line 199 "parser.y"
    { init((yyval), "variable");
               stack((yyval)).operands().swap(stack((yyvsp[(2) - (2)])).operands());
             }
    break;

  case 8:
#line 205 "parser.y"
    { init((yyval)); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 9:
#line 207 "parser.y"
    { (yyval)=(yyvsp[(1) - (3)]); mto((yyval), (yyvsp[(3) - (3)])); }
    break;

  case 10:
#line 211 "parser.y"
    { init((yyval)); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 11:
#line 213 "parser.y"
    { (yyval)=(yyvsp[(1) - (3)]); mto((yyval), (yyvsp[(3) - (3)])); }
    break;

  case 12:
#line 216 "parser.y"
    { init((yyval)); }
    break;

  case 14:
#line 220 "parser.y"
    { init((yyval)); }
    break;

  case 16:
#line 224 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); }
    break;

  case 17:
#line 227 "parser.y"
    { init((yyval), "empty"); }
    break;

  case 18:
#line 228 "parser.y"
    { init((yyval), "bool"); }
    break;

  case 19:
#line 230 "parser.y"
    {
               init((yyval), "bool-vector");
               stack((yyval)).set("width", stack((yyvsp[(3) - (4)])).id());
             }
    break;

  case 20:
#line 237 "parser.y"
    { PARSER.function=stack((yyvsp[(3) - (3)])).get("identifier"); }
    break;

  case 21:
#line 239 "parser.y"
    { PARSER.function=""; }
    break;

  case 22:
#line 240 "parser.y"
    { init((yyval), "function");
               stack((yyval)).add("return_type").swap(stack((yyvsp[(2) - (7)])));
               stack((yyval)).set("identifier", stack((yyvsp[(3) - (7)])).get("identifier"));
               stack((yyval)).add("arguments").get_sub().swap(
                 stack((yyvsp[(5) - (7)])).add("operands").get_sub());
               stack((yyval)).add("body").swap(stack((yyvsp[(6) - (7)])));
             }
    break;

  case 26:
#line 256 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); }
    break;

  case 27:
#line 259 "parser.y"
    { statement((yyval), "block"); }
    break;

  case 28:
#line 260 "parser.y"
    { (yyval)=(yyvsp[(1) - (2)]); }
    break;

  case 29:
#line 261 "parser.y"
    { (yyval)=(yyvsp[(1) - (1)]); }
    break;

  case 30:
#line 265 "parser.y"
    { (yyval)=(yyvsp[(1) - (3)]); mto((yyval), (yyvsp[(3) - (3)])); }
    break;

  case 31:
#line 266 "parser.y"
    { statement((yyval), "block"); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 32:
#line 270 "parser.y"
    { statement((yyval), "ifthenelse");
               stack((yyval)).move_to_operands(stack((yyvsp[(2) - (5)])), stack((yyvsp[(4) - (5)]))); }
    break;

  case 33:
#line 274 "parser.y"
    { statement((yyval), "ifthenelse");
               stack((yyval)).move_to_operands(stack((yyvsp[(2) - (7)])), stack((yyvsp[(4) - (7)])), stack((yyvsp[(6) - (7)]))); }
    break;

  case 34:
#line 279 "parser.y"
    { 
               if(stack((yyvsp[(2) - (2)])).operands().size()==1)
               {
                 statement((yyval), "goto");
                 stack((yyval)).set("destination", stack((yyvsp[(2) - (2)])).op0().id());
               }
               else
               {
                 statement((yyval), "non-deterministic-goto");
                 stack((yyval)).add("destinations").get_sub().swap(stack((yyvsp[(2) - (2)])).add("operands").get_sub());
               }
             }
    break;

  case 35:
#line 294 "parser.y"
    {
               init((yyval)); mto((yyval), (yyvsp[(1) - (1)]));
             }
    break;

  case 36:
#line 298 "parser.y"
    {
               (yyval)=(yyvsp[(1) - (3)]); mto((yyval), (yyvsp[(3) - (3)]));
             }
    break;

  case 37:
#line 304 "parser.y"
    { statement((yyval), "function_call");
               stack((yyval)).operands().resize(3);
               stack((yyval)).op0().make_nil();
               stack((yyval)).op1().swap(stack((yyvsp[(1) - (4)])));
               stack((yyval)).op2().id("arguments");
               stack((yyval)).op2().operands().swap(stack((yyvsp[(3) - (4)])).operands());
             }
    break;

  case 39:
#line 317 "parser.y"
    { statement((yyval), "return");
               stack((yyval)).operands().swap(stack((yyvsp[(2) - (2)])).operands());
             }
    break;

  case 40:
#line 320 "parser.y"
    { statement((yyval), "return"); }
    break;

  case 41:
#line 324 "parser.y"
    { statement((yyval), "bp_enforce");
               stack((yyval)).reserve_operands(2); // for code
               mto((yyval), (yyvsp[(2) - (2)]));
               stack((yyval)).operands().resize(2);
             }
    break;

  case 42:
#line 332 "parser.y"
    { statement((yyval), "skip");
               stack((yyval)).set("explicit", true);
             }
    break;

  case 43:
#line 338 "parser.y"
    { statement((yyval), "bp_abortif");
               mto((yyval), (yyvsp[(2) - (2)]));
             }
    break;

  case 44:
#line 344 "parser.y"
    { statement((yyval), "start_thread");
               mto((yyval), (yyvsp[(2) - (2)]));
             }
    break;

  case 45:
#line 350 "parser.y"
    { statement((yyval), "end_thread");
             }
    break;

  case 46:
#line 355 "parser.y"
    { statement((yyval), "sync");
               stack((yyval)).set("event", stack((yyvsp[(2) - (2)])).id());
             }
    break;

  case 47:
#line 361 "parser.y"
    { statement((yyval), "atomic_begin");
             }
    break;

  case 48:
#line 366 "parser.y"
    { statement((yyval), "atomic_end");
             }
    break;

  case 49:
#line 371 "parser.y"
    { statement((yyval), "bp_dead");
               stack((yyval)).operands().swap(stack((yyvsp[(2) - (2)])).operands());
             }
    break;

  case 66:
#line 395 "parser.y"
    { statement((yyval), "decl");
               stack((yyval)).operands().swap(stack((yyvsp[(1) - (1)])).operands());
             }
    break;

  case 67:
#line 401 "parser.y"
    {
               init((yyval));
               stack((yyval)).make_nil();
             }
    break;

  case 69:
#line 409 "parser.y"
    {
               (yyval)=(yyvsp[(2) - (2)]);
             }
    break;

  case 70:
#line 415 "parser.y"
    {
               statement((yyval), "assign");
               stack((yyval)).reserve_operands(2);
               mto((yyval), (yyvsp[(1) - (4)]));
               mto((yyval), (yyvsp[(3) - (4)]));

               if(stack((yyvsp[(4) - (4)])).is_not_nil())
               {
                 exprt tmp;
                 tmp.swap(stack((yyval)));
                 
                 init(stack((yyval)));
                 stack((yyval)).id("code");
                 stack((yyval)).set("statement", "bp_constrain");
                 stack((yyval)).move_to_operands(tmp, stack((yyvsp[(4) - (4)])));
               }
             }
    break;

  case 71:
#line 433 "parser.y"
    {
               (yyval)=(yyvsp[(3) - (3)]);
               stack((yyval)).op0().swap(stack((yyvsp[(1) - (3)])));
             }
    break;

  case 72:
#line 440 "parser.y"
    {
               statement((yyval), "label");
               stack((yyval)).set("label", stack((yyvsp[(1) - (3)])).id());
               mto((yyval), (yyvsp[(3) - (3)]));
             }
    break;

  case 74:
#line 451 "parser.y"
    { init((yyval), "symbol");
               stack((yyval)).set("identifier", stack((yyvsp[(1) - (1)])).id());
             }
    break;

  case 75:
#line 457 "parser.y"
    { init((yyval), "symbol");
               stack((yyval)).set("identifier", stack((yyvsp[(1) - (1)])).id());
             }
    break;

  case 77:
#line 464 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "=>", (yyvsp[(3) - (3)])); }
    break;

  case 78:
#line 465 "parser.y"
    { j_binary((yyval), (yyvsp[(1) - (3)]), "and", (yyvsp[(3) - (3)])); }
    break;

  case 79:
#line 466 "parser.y"
    { j_binary((yyval), (yyvsp[(1) - (3)]), "or", (yyvsp[(3) - (3)])); }
    break;

  case 80:
#line 467 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "xor", (yyvsp[(3) - (3)])); }
    break;

  case 81:
#line 468 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "=", (yyvsp[(3) - (3)])); }
    break;

  case 82:
#line 469 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "notequal", (yyvsp[(3) - (3)])); }
    break;

  case 83:
#line 472 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); }
    break;

  case 84:
#line 473 "parser.y"
    { init((yyval), "nondet_bool"); }
    break;

  case 86:
#line 476 "parser.y"
    {
               init((yyval), "bp_schoose");
               stack((yyval)).operands().swap(stack((yyvsp[(3) - (4)])).operands());
             }
    break;

  case 87:
#line 481 "parser.y"
    {
               init((yyval), "constant");
               stack((yyval)).set("value", stack((yyvsp[(1) - (1)])).id());
             }
    break;

  case 89:
#line 489 "parser.y"
    { init((yyval), "not"); mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 90:
#line 490 "parser.y"
    { init((yyval), "not"); mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 91:
#line 491 "parser.y"
    { init((yyval), "tick"); mto((yyval), (yyvsp[(2) - (2)])); }
    break;


/* Line 1267 of yacc.c.  */
#line 2088 "y.tab.cpp"
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


#line 494 "parser.y"


int yybperror(const std::string &error)
{
  PARSER.parse_error(error, yytext);
  return 0;
}

#undef yyerror


