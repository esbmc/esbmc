/* A Bison parser, made by GNU Bison 2.4.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
   2009, 2010 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

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




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yybplval;


