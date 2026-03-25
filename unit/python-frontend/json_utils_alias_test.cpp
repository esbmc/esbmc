#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <nlohmann/json.hpp>

#include <python-frontend/json_utils.h>

TEST_CASE("json_utils get_object_alias resolves import aliases", "[python-frontend][json-utils]")
{
  nlohmann::json ast = {
    {"body",
     nlohmann::json::array(
       {{{"_type", "ImportFrom"},
         {"module", "math"},
         {"names",
          nlohmann::json::array(
            {{{"_type", "alias"}, {"name", "sqrt"}, {"asname", "sq"}}})}}})}};

  REQUIRE(json_utils::get_object_alias(ast, "sq") == "sqrt");
  REQUIRE(json_utils::get_object_alias(ast, "sqrt") == "sqrt");
}

TEST_CASE("json_utils get_object_alias falls back when no alias exists", "[python-frontend][json-utils]")
{
  nlohmann::json ast = {{"body", nlohmann::json::array()}};

  REQUIRE(json_utils::get_object_alias(ast, "module.func") == "module.func");
  REQUIRE(json_utils::get_object_alias(ast, "name") == "name");
}
