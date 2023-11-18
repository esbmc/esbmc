
#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <catch2/catch.hpp>
#include <python-frontend/python_annotation.h>
#include <nlohmann/json.hpp>

TEST_CASE("Add type annotation")
{
  SECTION("Get type from constant")
  {
    std::istringstream input_data(R"json([
            {
                "_type": "Assign",
                "col_offset": 0,
                "end_col_offset": 6,
                "end_lineno": 1,
                "lineno": 1,
                "targets": [
                    {
                        "_type": "Name",
                        "col_offset": 0,
                        "ctx": {
                            "_type": "Store"
                        },
                        "end_col_offset": 1,
                        "end_lineno": 1,
                        "id": "n",
                        "lineno": 1
                    }
                ],
                "type_comment": null,
                "value": {
                    "_type": "Constant",
                    "col_offset": 4,
                    "end_col_offset": 6,
                    "end_lineno": 1,
                    "kind": null,
                    "lineno": 1,
                    "n": 10,
                    "s": 10,
                    "value": 10
                }
            }
        ])json");

    nlohmann::json input_json;
    input_data >> input_json;

    std::istringstream output_data(R"json([
                {
                    "_type": "AnnAssign",
                    "annotation": {
                        "_type": "Name",
                        "col_offset": 2,
                        "ctx": {
                            "_type": "Load"
                        },
                        "end_col_offset": 5,
                        "end_lineno": 1,
                        "id": "int",
                        "lineno": 1
                    },
                    "col_offset": 0,
                    "end_col_offset": 10,
                    "end_lineno": 1,
                    "lineno": 1,
                    "simple": 1,
                    "target": {
                        "_type": "Name",
                        "col_offset": 0,
                        "ctx": {
                            "_type": "Store"
                        },
                        "end_col_offset": 1,
                        "end_lineno": 1,
                        "id": "n",
                        "lineno": 1
                    },
                    "type_comment": null,
                    "value": {
                        "_type": "Constant",
                        "col_offset": 8,
                        "end_col_offset": 10,
                        "end_lineno": 1,
                        "kind": null,
                        "lineno": 1,
                        "n": 10,
                        "s": 10,
                        "value": 10
                    }
                }
            ])json");

    nlohmann::json expected_output;
	output_data >> expected_output;

    python_annotation<nlohmann::json> ann;
    ann.add_type_annotation(input_json);

    REQUIRE(input_json == expected_output);
  }
}
