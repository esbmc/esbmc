/*******************************************************************
 Module: Builtin Libs unit test

 Author: Rafael Sá Menezes

 Date: December 2020

 Test Plan:
   - Functions (Sync)
   - Functions (Async - Pthread)
 \*******************************************************************/

#define BOOST_TEST_MODULE "Builtin Models"

#include <buitin_libs.hs>
#include <boost/test/included/unit_test.hpp>


void __ESBMC_atomic_begin() {}
void __ESBMC_atomic_end() {}


namespace utf = boost::unit_test;

#define sync_fetch_generator(TYPE, OPERATOR) \
    {int dest = 10;                     \
    TYPE value = 5;                          \
    int fetch = __builtin_esbmc_sync_fetch_and_##OPERATOR(&dest, value); \
    BOOST_TEST(dest == 15);                  \
    BOOST_TEST(fetch == 10);}

BOOST_AUTO_TEST_SUITE(functions_sync)

BOOST_AUTO_TEST_CASE(sync_fetch_add)
{
  sync_fetch_generator(int, add);
  sync_fetch_generator(double, add);
  sync_fetch_generator(float, add);
  sync_fetch_generator(short, add);
  sync_fetch_generator(char, add);
}
BOOST_AUTO_TEST_SUITE_END()

