/*******************************************************************\
Module: Unit tests for filesystem_operations class
Author: Rafael Sá Menezes

\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/filesystem.h>

TEST_CASE(
          "tmp folder should be unique between two runs", "[core][util][filesystem]")     
{
  const char *format = "esbmc-test-%%%%";
auto first = file_operations::get_unique_tmp_path(format);
auto second = file_operations::get_unique_tmp_path(format);
REQUIRE(first != second);
}
