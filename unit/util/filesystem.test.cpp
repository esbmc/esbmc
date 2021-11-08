/*******************************************************************\
Module: Unit tests for filesystem_operations class
Author: Rafael Sá Menezes

\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/filesystem.h>
#include <boost/filesystem.hpp>

TEST_CASE(
  "tmp path should be unique between two runs",
  "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  auto first = file_operations::get_unique_tmp_path(format);
  auto second = file_operations::get_unique_tmp_path(format);
  REQUIRE(first != second);
}

TEST_CASE(
  "tmp folder should be unique between two runs",
  "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  auto first = file_operations::create_tmp_dir(format);
  auto second = file_operations::create_tmp_dir(format);
  REQUIRE(first.path() != second.path());
}

TEST_CASE(
  "tmp file should be unique between two runs",
  "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  auto first = file_operations::create_tmp_file(format);
  auto second = file_operations::create_tmp_file(format);
  REQUIRE(first.path() != second.path());
}

TEST_CASE("tmp dir is dir and should be removed", "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  std::string path;
  {
    auto dir = file_operations::create_tmp_dir(format);
    path = dir.path();
    REQUIRE(boost::filesystem::is_directory(path));
  }
  REQUIRE(!boost::filesystem::exists(path));
}

TEST_CASE("tmp file is file and should be removed", "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  std::string path;
  {
    auto file = file_operations::create_tmp_file(format);
    path = file.path();
    REQUIRE(boost::filesystem::is_regular_file(path));
  }
  REQUIRE(!boost::filesystem::exists(path));
}
