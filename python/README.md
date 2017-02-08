# ESBMC python bindings

Contents:
 * Introduction
 * Building
 * Caveat emptor
 * Runtime examination and debugging
 * Use as a library
 * Caveat programmer
 * Future directions
 * Even more caveats

## Introduction

The ESBMC python bindings allow access to the internal APIs and data structures
of ESBMC, without having to write any native code or depend on headers. The
entire system is contained in a single shared object. A replication of the ESBMC
command line program can be achieved in roughly thirty lines of python. Most
importantly, python code can be hooked into ESBMC at various stages, allowing 
inspection and extension. We have chosen python because:
 * It's a great language
 * Rapid prototyping and even heavy lifting is easy to achieve
 * There's a wealth of libraries already available, easily interoperability
   between BMC and other technology
 * The selected binding mechanism (Boost.Python) is (largely) type safe and
   directly encapsulates our existing C++ facilities

It's important to understand that these python bindings are just that: an API
for accessing the underlying C++ API. Thus if you perform illegal behaviour
with the API such as an out-of-range vector access, you'll still likely
experience a crash. The benefits comes from being able to compose objects /
algorithms / data structures from python with ESBMC objects.

Accordingly, these bindings come with some hard truths: using them is not a
substitute for knowing about (much of) ESBMCs internals, you still have to
understand what you're dealing with. They don't provide anything more than
what ESBMC already does. They are no silver bullet for making an analysis or
extension work (they just make it easier). Finally, the moment you touch python
you immediately pay a performance penalty.

## Building

 * Requires python-dev, of course
 * That allows for python instances within ESBMC process
 * Must be shared object to be used in any other python process
 * --enable-shared
 * Env vars
 * Boost python

## Caveat emptor

 * As mentioned, no substitude for knowledge
 * Some things aren't exported
 * Some thingis can't be exported easily
 * Python isn't going to slow down C++ development
 * Not the default

## Runtime examination and debugging

 * Can break in and have RT object in locals
 * Might be able to use this to instrument various points in ESBMC
 * Lists
 * Dicts

## Use as a library

 * ESBMC contains globals
 * One instance per process pls
 * Basic outline of the high level point that exporting stops
 * If you want to over-ride internals, you might need to implement up to that
   point
 * Might be addressed in the future via factories
 * Might also provide a high level python library that does the main driving
 * Solver API as an example?

## Caveat programmer

 * Boost.python is crazy shit
 * Reference counting
 * Downcasting, lol, and other gotchas
 * I don't really intend on making things "nice" for python, i.e. irep\_idt 
   as\_string
 * Wrapped objects?
 * Const correctness
 * XXX that compiler bug for irep2?
 * Exceptions through a C++ call stack might become fairly crazy
 * 'None' and when things are non-evalued internal references

## Future directions

 * No idea

## Even more caveats

 * Best to assume that esbmc is going to crash every time you do something
 * boost shared ptrs?
 * I subclassed an stdl container
 * old irep?
 * Probably can't extend things like enums
