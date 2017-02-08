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

To build these bindings you need to have your distributions python 3 development
package installed, which the configure script should automatically discover. You
also need Boost.Python to be installed, it's most likely in your distributions
libboost package.

The configure script is hard coded to build for python 3: technically the
autoconf file can be edited to build against python 2, and Boost.Python will
work just as well with that. At the time of writing, python 2 only has three
years of life left, you should avoid starting new projects in it.

Two switches to the configure script are required to build the python bindings:
 * --enable-python
 * --enable-shared
The former is self explanatory, the latter causes a shared object (.so) library
of ESBMC functions to be build. If you're not interested in the ESBMC command
line too, you might want to add --disable-esbmc to the configure command line,
which will avoid building static object files, and will save you 50% of build
time.

Once built, a libesbmc.so.0.0.0 file will exist in the esbmc/.libs directory of
your build directory. To access ESBMC from python, symlink (or copy) the shared
object into your python path as 'esbmc.so', and import it from python:

    import esbmc

We might get around to writing some examples; in the mean time there should be
some regression tests published that illustrate accessing ESBMC apis from
python.

## Caveat emptor

As mentioned above, these bindings are not a substitute for knowing about ESBMCs
internals, and there are limitations as to what can be done. There are no good
python representations of, for example, list iterators that are used extensively
throughout ESBMC instead of pointers. If you need to deeply mess with internals,
you will need to use C++.

Certain APIs are not exported from ESBMC to python, for example fiddly
intrinsics, everything to do with parsing and GOTO conversion, and things like
state hashing. The most likely reasons for a facility not being exported is that
it's not particularly useful as a library or interesting, or that it has a very
complicated type signature.

We make no attempt to preserve a stable ABI or API between ESBMC releases, and
as a result all releases are breaking releases. The good news is that python
insulates against ABI changes, and you'll only experience exceptions if a
function signature changes (or is deleted), which should always present when you
call into ESBMC from python.

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
