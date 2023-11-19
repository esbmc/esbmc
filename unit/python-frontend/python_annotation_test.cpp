
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

    python_annotation<nlohmann::json> ann(input_json);
    ann.add_type_annotation();

    REQUIRE(input_json == expected_output);
  }

  SECTION("Get LHS type from RHS type")
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
        },
        {
            "_type": "Assign",
            "col_offset": 0,
            "end_col_offset": 5,
            "end_lineno": 2,
            "lineno": 2,
            "targets": [
                {
                    "_type": "Name",
                    "col_offset": 0,
                    "ctx": {
                        "_type": "Store"
                    },
                    "end_col_offset": 1,
                    "end_lineno": 2,
                    "id": "p",
                    "lineno": 2
                }
            ],
            "type_comment": null,
            "value": {
                "_type": "Name",
                "col_offset": 4,
                "ctx": {
                    "_type": "Load"
                },
                "end_col_offset": 5,
                "end_lineno": 2,
                "id": "n",
                "lineno": 2
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
        },
        {
            "_type": "AnnAssign",
            "annotation": {
                "_type": "Name",
                "col_offset": 2,
                "ctx": {
                    "_type": "Load"
                },
                "end_col_offset": 5,
                "end_lineno": 2,
                "id": "int",
                "lineno": 2
            },
            "col_offset": 0,
            "end_col_offset": 9,
            "end_lineno": 2,
            "lineno": 2,
            "simple": 1,
            "target": {
                "_type": "Name",
                "col_offset": 0,
                "ctx": {
                    "_type": "Store"
                },
                "end_col_offset": 1,
                "end_lineno": 2,
                "id": "p",
                "lineno": 2
            },
            "value": {
                "_type": "Name",
                "col_offset": 8,
                "ctx": {
                    "_type": "Load"
                },
                "end_col_offset": 9,
                "end_lineno": 2,
                "id": "n",
                "lineno": 2
            }
        }
    ])json");

    nlohmann::json expected_output;
    output_data >> expected_output;

    python_annotation<nlohmann::json> ann(input_json);
    ann.add_type_annotation();

    REQUIRE(input_json == expected_output);
  }

  SECTION("Get LHS type in function body")
  {
    std::istringstream input_data(R"json([
        {
            "_type": "FunctionDef",
            "args": {
                "_type": "arguments",
                "args": [],
                "defaults": [],
                "kw_defaults": [],
                "kwarg": null,
                "kwonlyargs": [],
                "posonlyargs": [],
                "vararg": null
            },
            "body": [
                {
                    "_type": "Assign",
                    "col_offset": 2,
                    "end_col_offset": 8,
                    "end_lineno": 5,
                    "lineno": 5,
                    "targets": [
                        {
                            "_type": "Name",
                            "col_offset": 2,
                            "ctx": {
                                "_type": "Store"
                            },
                            "end_col_offset": 3,
                            "end_lineno": 5,
                            "id": "b",
                            "lineno": 5
                        }
                    ],
                    "type_comment": null,
                    "value": {
                        "_type": "Constant",
                        "col_offset": 6,
                        "end_col_offset": 8,
                        "end_lineno": 5,
                        "kind": null,
                        "lineno": 5,
                        "n": 10,
                        "s": 10,
                        "value": 10
                    }
                }
            ],
            "col_offset": 0,
            "decorator_list": [],
            "end_col_offset": 8,
            "end_lineno": 5,
            "lineno": 4,
            "name": "foo",
            "returns": {
                "_type": "Constant",
                "col_offset": 13,
                "end_col_offset": 17,
                "end_lineno": 4,
                "kind": null,
                "lineno": 4,
                "n": null,
                "s": null,
                "value": null
            },
            "type_comment": null
        }
    ])json");

    nlohmann::json input_json;
    input_data >> input_json;

    std::istringstream output_data(R"json([
        {
            "_type": "FunctionDef",
            "args": {
                "_type": "arguments",
                "args": [],
                "defaults": [],
                "kw_defaults": [],
                "kwarg": null,
                "kwonlyargs": [],
                "posonlyargs": [],
                "vararg": null
            },
            "body": [
                {
                    "_type": "AnnAssign",
                    "annotation": {
                        "_type": "Name",
                        "col_offset": 4,
                        "ctx": {
                            "_type": "Load"
                        },
                        "end_col_offset": 7,
                        "end_lineno": 5,
                        "id": "int",
                        "lineno": 5
                    },
                    "col_offset": 2,
                    "end_col_offset": 12,
                    "end_lineno": 5,
                    "lineno": 5,
                    "simple": 1,
                    "target": {
                        "_type": "Name",
                        "col_offset": 2,
                        "ctx": {
                            "_type": "Store"
                        },
                        "end_col_offset": 3,
                        "end_lineno": 5,
                        "id": "b",
                        "lineno": 5
                    },
                    "value": {
                        "_type": "Constant",
                        "col_offset": 10,
                        "end_col_offset": 12,
                        "end_lineno": 5,
                        "kind": null,
                        "lineno": 5,
                        "n": 10,
                        "s": 10,
                        "value": 10
                    }
                }
            ],
            "col_offset": 0,
            "decorator_list": [],
            "end_col_offset": 12,
            "end_lineno": 5,
            "lineno": 4,
            "name": "foo",
            "returns": {
                "_type": "Constant",
                "col_offset": 13,
                "end_col_offset": 17,
                "end_lineno": 4,
                "kind": null,
                "lineno": 4,
                "n": null,
                "s": null,
                "value": null
            },
            "type_comment": null
        }
    ])json");

    nlohmann::json expected_output;
    output_data >> expected_output;

    python_annotation<nlohmann::json> ann(input_json);
    ann.add_type_annotation();

    //REQUIRE(input_json == expected_output);
  }
}
