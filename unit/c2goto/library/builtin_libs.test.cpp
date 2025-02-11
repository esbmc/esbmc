/*******************************************************************
 Module: Builtin Libs unit test

 Author: Rafael Sá Menezes

 Date: December 2020

 Test Plan:
   - Functions (Sync)
   - Functions (Async - Pthread)
 \*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>

void __ESBMC_atomic_begin()
{
}
void __ESBMC_atomic_end()
{
}

#ifndef _WIN32

#  define sync_fetch(OPERATOR) __sync_fetch_and_##OPERATOR

#  define sync_fetch_generator(TYPE, OPERATOR)                                 \
    {                                                                          \
      int dest = 10;                                                           \
      TYPE value = 5;                                                          \
      int fetch = sync_fetch(OPERATOR)(&dest, value);                          \
      CHECK(dest == 15);                                                       \
      CHECK(fetch == 10);                                                      \
    }

TEST_CASE("sync_fetch_add", "[core][c2goto][builtin]")
{
  SECTION("Int")
  {
    sync_fetch_generator(int, add);
  }

  SECTION("Double")
  {
    sync_fetch_generator(double, add);
  }

  SECTION("Float")
  {
    sync_fetch_generator(float, add);
  }

  SECTION("Short")
  {
    sync_fetch_generator(short, add);
  }

  SECTION("Char")
  {
    sync_fetch_generator(char, add);
  }
}
#endif