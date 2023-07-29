#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include "goto-programs/abstract-interpretation/common_subexpression_elimination.h"

struct test_item
{
  expr2tc e;
  bool should_contain;
};


TEST_CASE("Hello", "[ai][available-expressions]")
{
}
