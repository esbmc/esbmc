
#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <catch2/catch.hpp>
#include <python-frontend/python_annotation.h>
#include <python-frontend/global_scope.h>
#include <nlohmann/json.hpp>

TEST_CASE("Add type annotation")
{
  SECTION("Get type from constant")
  {
    std::istringstream input_data(R"json({
      "_type": "Module",
      "body": [
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
      ],
      "filename": "test.py",
      "type_ignores": []
     })json");

    nlohmann::json input_json;
    input_data >> input_json;

    std::istringstream output_data(R"json({
        "_type": "Module",
        "body": [
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
        ],
        "filename": "test.py",
        "type_ignores": []
      })json");

    nlohmann::json expected_output;
    output_data >> expected_output;

    global_scope gs;
    python_annotation<nlohmann::json> ann(input_json, gs);
    ann.add_type_annotation();

    REQUIRE(input_json == expected_output);
  }

  SECTION("Get LHS type from RHS type")
  {
    std::istringstream input_data(R"json({
    "_type": "Module",
    "body": [
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
    ],
    "filename": "test.py",
    "type_ignores": []
    })json");

    nlohmann::json input_json;
    input_data >> input_json;

    std::istringstream output_data(R"json({
    "_type": "Module",
    "body": [
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
    ],
    "filename": "test.py",
    "type_ignores": []
    })json");

    nlohmann::json expected_output;
    output_data >> expected_output;

    global_scope gs;
    python_annotation<nlohmann::json> ann(input_json, gs);
    ann.add_type_annotation();

    REQUIRE(input_json == expected_output);
  }

  SECTION("Get LHS type in function body")
  {
    std::istringstream input_data(R"json({
    "_type": "Module",
    "body": [
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
                    "end_lineno": 2,
                    "lineno": 2,
                    "targets": [
                        {
                            "_type": "Name",
                            "col_offset": 2,
                            "ctx": {
                                "_type": "Store"
                            },
                            "end_col_offset": 3,
                            "end_lineno": 2,
                            "id": "n",
                            "lineno": 2
                        }
                    ],
                    "type_comment": null,
                    "value": {
                        "_type": "Constant",
                        "col_offset": 6,
                        "end_col_offset": 8,
                        "end_lineno": 2,
                        "kind": null,
                        "lineno": 2,
                        "n": 10,
                        "s": 10,
                        "value": 10
                    }
                }
            ],
            "col_offset": 0,
            "decorator_list": [],
            "end_col_offset": 8,
            "end_lineno": 2,
            "lineno": 1,
            "name": "foo",
            "returns": {
                "_type": "Constant",
                "col_offset": 13,
                "end_col_offset": 17,
                "end_lineno": 1,
                "kind": null,
                "lineno": 1,
                "n": null,
                "s": null,
                "value": null
            },
            "type_comment": null
        }
    ],
    "filename": "test.py",
    "type_ignores": []
    })json");

    nlohmann::json input_json;
    input_data >> input_json;

    std::istringstream output_data(R"json({
    "_type": "Module",
    "body": [
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
                        "end_lineno": 2,
                        "id": "int",
                        "lineno": 2
                    },
                    "col_offset": 2,
                    "end_col_offset": 12,
                    "end_lineno": 2,
                    "lineno": 2,
                    "simple": 1,
                    "target": {
                        "_type": "Name",
                        "col_offset": 2,
                        "ctx": {
                            "_type": "Store"
                        },
                        "end_col_offset": 3,
                        "end_lineno": 2,
                        "id": "n",
                        "lineno": 2
                    },
                    "value": {
                        "_type": "Constant",
                        "col_offset": 10,
                        "end_col_offset": 12,
                        "end_lineno": 2,
                        "kind": null,
                        "lineno": 2,
                        "n": 10,
                        "s": 10,
                        "value": 10
                    }
                }
            ],
            "col_offset": 0,
            "decorator_list": [],
            "end_col_offset": 12,
            "end_lineno": 2,
            "lineno": 1,
            "name": "foo",
            "returns": {
                "_type": "Constant",
                "col_offset": 13,
                "end_col_offset": 17,
                "end_lineno": 1,
                "kind": null,
                "lineno": 1,
                "n": null,
                "s": null,
                "value": null
            },
            "type_comment": null
        }
    ],
    "filename": "test.py",
    "type_ignores": []
    })json");

    nlohmann::json expected_output;
    output_data >> expected_output;

    global_scope gs;
    python_annotation<nlohmann::json> ann(input_json, gs);
    ann.add_type_annotation();

    REQUIRE(input_json == expected_output);
  }

  SECTION("Get LHS type from function args")
  {
    std::istringstream input_data(R"json({
    "_type": "Module",
    "body": [
        {
            "_type": "FunctionDef",
            "args": {
                "_type": "arguments",
                "args": [
                    {
                        "_type": "arg",
                        "annotation": {
                            "_type": "Name",
                            "col_offset": 10,
                            "ctx": {
                                "_type": "Load"
                            },
                            "end_col_offset": 13,
                            "end_lineno": 1,
                            "id": "int",
                            "lineno": 1
                        },
                        "arg": "a",
                        "col_offset": 8,
                        "end_col_offset": 13,
                        "end_lineno": 1,
                        "lineno": 1,
                        "type_comment": null
                    }
                ],
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
                    "end_col_offset": 7,
                    "end_lineno": 2,
                    "lineno": 2,
                    "targets": [
                        {
                            "_type": "Name",
                            "col_offset": 2,
                            "ctx": {
                                "_type": "Store"
                            },
                            "end_col_offset": 3,
                            "end_lineno": 2,
                            "id": "b",
                            "lineno": 2
                        }
                    ],
                    "type_comment": null,
                    "value": {
                        "_type": "Name",
                        "col_offset": 6,
                        "ctx": {
                            "_type": "Load"
                        },
                        "end_col_offset": 7,
                        "end_lineno": 2,
                        "id": "a",
                        "lineno": 2
                    }
                }
            ],
            "col_offset": 0,
            "decorator_list": [],
            "end_col_offset": 7,
            "end_lineno": 2,
            "lineno": 1,
            "name": "foo",
            "returns": {
                "_type": "Constant",
                "col_offset": 18,
                "end_col_offset": 22,
                "end_lineno": 1,
                "kind": null,
                "lineno": 1,
                "n": null,
                "s": null,
                "value": null
            },
            "type_comment": null
        }
    ],
    "filename": "test.py",
    "type_ignores": []
    })json");

    nlohmann::json input_json;
    input_data >> input_json;

    std::istringstream output_data(R"json({
    "_type": "Module",
    "body": [
        {
            "_type": "FunctionDef",
            "args": {
                "_type": "arguments",
                "args": [
                    {
                        "_type": "arg",
                        "annotation": {
                            "_type": "Name",
                            "col_offset": 10,
                            "ctx": {
                                "_type": "Load"
                            },
                            "end_col_offset": 13,
                            "end_lineno": 1,
                            "id": "int",
                            "lineno": 1
                        },
                        "arg": "a",
                        "col_offset": 8,
                        "end_col_offset": 13,
                        "end_lineno": 1,
                        "lineno": 1,
                        "type_comment": null
                    }
                ],
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
                        "end_lineno": 2,
                        "id": "int",
                        "lineno": 2
                    },
                    "col_offset": 2,
                    "end_col_offset": 11,
                    "end_lineno": 2,
                    "lineno": 2,
                    "simple": 1,
                    "target": {
                        "_type": "Name",
                        "col_offset": 2,
                        "ctx": {
                            "_type": "Store"
                        },
                        "end_col_offset": 3,
                        "end_lineno": 2,
                        "id": "b",
                        "lineno": 2
                    },
                    "value": {
                        "_type": "Name",
                        "col_offset": 10,
                        "ctx": {
                            "_type": "Load"
                        },
                        "end_col_offset": 11,
                        "end_lineno": 2,
                        "id": "a",
                        "lineno": 2
                    }
                }
            ],
            "col_offset": 0,
            "decorator_list": [],
            "end_col_offset": 11,
            "end_lineno": 2,
            "lineno": 1,
            "name": "foo",
            "returns": {
                "_type": "Constant",
                "col_offset": 18,
                "end_col_offset": 22,
                "end_lineno": 1,
                "kind": null,
                "lineno": 1,
                "n": null,
                "s": null,
                "value": null
            },
            "type_comment": null
        }
    ],
    "filename": "test.py",
    "type_ignores": []
    })json");

    nlohmann::json expected_output;
    output_data >> expected_output;

    global_scope gs;
    python_annotation<nlohmann::json> ann(input_json, gs);
    ann.add_type_annotation();

    REQUIRE(input_json == expected_output);
  }

  SECTION("Get LHS type from constructor call")
  {
    std::istringstream input_data(R"json({
    "_type": "Module",
    "body": [
        {
            "_type": "ClassDef",
            "bases": [],
            "body": [
                {
                    "_type": "FunctionDef",
                    "args": {
                        "_type": "arguments",
                        "args": [
                            {
                                "_type": "arg",
                                "annotation": null,
                                "arg": "self",
                                "col_offset": 17,
                                "end_col_offset": 21,
                                "end_lineno": 2,
                                "lineno": 2,
                                "type_comment": null
                            },
                            {
                                "_type": "arg",
                                "annotation": {
                                    "_type": "Name",
                                    "col_offset": 30,
                                    "ctx": {
                                        "_type": "Load"
                                    },
                                    "end_col_offset": 33,
                                    "end_lineno": 2,
                                    "id": "int",
                                    "lineno": 2
                                },
                                "arg": "value",
                                "col_offset": 23,
                                "end_col_offset": 33,
                                "end_lineno": 2,
                                "lineno": 2,
                                "type_comment": null
                            }
                        ],
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
                                "col_offset": 18,
                                "ctx": {
                                    "_type": "Load"
                                },
                                "end_col_offset": 21,
                                "end_lineno": 3,
                                "id": "int",
                                "lineno": 3
                            },
                            "col_offset": 8,
                            "end_col_offset": 29,
                            "end_lineno": 3,
                            "lineno": 3,
                            "simple": 0,
                            "target": {
                                "_type": "Attribute",
                                "attr": "blah",
                                "col_offset": 8,
                                "ctx": {
                                    "_type": "Store"
                                },
                                "end_col_offset": 17,
                                "end_lineno": 3,
                                "lineno": 3,
                                "value": {
                                    "_type": "Name",
                                    "col_offset": 8,
                                    "ctx": {
                                        "_type": "Load"
                                    },
                                    "end_col_offset": 12,
                                    "end_lineno": 3,
                                    "id": "self",
                                    "lineno": 3
                                }
                            },
                            "value": {
                                "_type": "Name",
                                "col_offset": 24,
                                "ctx": {
                                    "_type": "Load"
                                },
                                "end_col_offset": 29,
                                "end_lineno": 3,
                                "id": "value",
                                "lineno": 3
                            }
                        }
                    ],
                    "col_offset": 4,
                    "decorator_list": [],
                    "end_col_offset": 29,
                    "end_lineno": 3,
                    "lineno": 2,
                    "name": "__init__",
                    "returns": null,
                    "type_comment": null
                }
            ],
            "col_offset": 0,
            "decorator_list": [],
            "end_col_offset": 29,
            "end_lineno": 3,
            "keywords": [],
            "lineno": 1,
            "name": "Foo"
        },
        {
            "_type": "Assign",
            "col_offset": 0,
            "end_col_offset": 10,
            "end_lineno": 6,
            "lineno": 6,
            "targets": [
                {
                    "_type": "Name",
                    "col_offset": 0,
                    "ctx": {
                        "_type": "Store"
                    },
                    "end_col_offset": 1,
                    "end_lineno": 6,
                    "id": "f",
                    "lineno": 6
                }
            ],
            "type_comment": null,
            "value": {
                "_type": "Call",
                "args": [
                    {
                        "_type": "Constant",
                        "col_offset": 8,
                        "end_col_offset": 9,
                        "end_lineno": 6,
                        "kind": null,
                        "lineno": 6,
                        "n": 0,
                        "s": 0,
                        "value": 0
                    }
                ],
                "col_offset": 4,
                "end_col_offset": 10,
                "end_lineno": 6,
                "func": {
                    "_type": "Name",
                    "col_offset": 4,
                    "ctx": {
                        "_type": "Load"
                    },
                    "end_col_offset": 7,
                    "end_lineno": 6,
                    "id": "Foo",
                    "lineno": 6
                },
                "keywords": [],
                "lineno": 6
            }
        }
    ],
    "filename": "main.py",
    "type_ignores": []
    })json");

    nlohmann::json input_json;
    input_data >> input_json;

    std::istringstream output_data(R"json({
    "_type": "Module",
    "body": [
        {
            "_type": "ClassDef",
            "bases": [],
            "body": [
                {
                    "_type": "FunctionDef",
                    "args": {
                        "_type": "arguments",
                        "args": [
                            {
                                "_type": "arg",
                                "annotation": null,
                                "arg": "self",
                                "col_offset": 17,
                                "end_col_offset": 21,
                                "end_lineno": 2,
                                "lineno": 2,
                                "type_comment": null
                            },
                            {
                                "_type": "arg",
                                "annotation": {
                                    "_type": "Name",
                                    "col_offset": 30,
                                    "ctx": {
                                        "_type": "Load"
                                    },
                                    "end_col_offset": 33,
                                    "end_lineno": 2,
                                    "id": "int",
                                    "lineno": 2
                                },
                                "arg": "value",
                                "col_offset": 23,
                                "end_col_offset": 33,
                                "end_lineno": 2,
                                "lineno": 2,
                                "type_comment": null
                            }
                        ],
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
                                "col_offset": 18,
                                "ctx": {
                                    "_type": "Load"
                                },
                                "end_col_offset": 21,
                                "end_lineno": 3,
                                "id": "int",
                                "lineno": 3
                            },
                            "col_offset": 8,
                            "end_col_offset": 29,
                            "end_lineno": 3,
                            "lineno": 3,
                            "simple": 0,
                            "target": {
                                "_type": "Attribute",
                                "attr": "blah",
                                "col_offset": 8,
                                "ctx": {
                                    "_type": "Store"
                                },
                                "end_col_offset": 17,
                                "end_lineno": 3,
                                "lineno": 3,
                                "value": {
                                    "_type": "Name",
                                    "col_offset": 8,
                                    "ctx": {
                                        "_type": "Load"
                                    },
                                    "end_col_offset": 12,
                                    "end_lineno": 3,
                                    "id": "self",
                                    "lineno": 3
                                }
                            },
                            "value": {
                                "_type": "Name",
                                "col_offset": 24,
                                "ctx": {
                                    "_type": "Load"
                                },
                                "end_col_offset": 29,
                                "end_lineno": 3,
                                "id": "value",
                                "lineno": 3
                            }
                        }
                    ],
                    "col_offset": 4,
                    "decorator_list": [],
                    "end_col_offset": 29,
                    "end_lineno": 3,
                    "lineno": 2,
                    "name": "__init__",
                    "returns": null,
                    "type_comment": null
                }
            ],
            "col_offset": 0,
            "decorator_list": [],
            "end_col_offset": 29,
            "end_lineno": 3,
            "keywords": [],
            "lineno": 1,
            "name": "Foo"
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
                "end_lineno": 6,
                "id": "Foo",
                "lineno": 6
            },
            "col_offset": 0,
            "end_col_offset": 14,
            "end_lineno": 6,
            "lineno": 6,
            "simple": 1,
            "target": {
                "_type": "Name",
                "col_offset": 0,
                "ctx": {
                    "_type": "Store"
                },
                "end_col_offset": 1,
                "end_lineno": 6,
                "id": "f",
                "lineno": 6
            },
            "value": {
                "_type": "Call",
                "args": [
                    {
                        "_type": "Constant",
                        "col_offset": 12,
                        "end_col_offset": 13,
                        "end_lineno": 6,
                        "kind": null,
                        "lineno": 6,
                        "n": 0,
                        "s": 0,
                        "value": 0
                    }
                ],
                "col_offset": 8,
                "end_col_offset": 14,
                "end_lineno": 6,
                "func": {
                    "_type": "Name",
                    "col_offset": 8,
                    "ctx": {
                        "_type": "Load"
                    },
                    "end_col_offset": 11,
                    "end_lineno": 6,
                    "id": "Foo",
                    "lineno": 6
                },
                "keywords": [],
                "lineno": 6
            }
        }
    ],
    "filename": "main.py",
    "type_ignores": []
    })json");

    nlohmann::json expected_output;
    output_data >> expected_output;

    global_scope gs;
    python_annotation<nlohmann::json> ann(input_json, gs);
    ann.add_type_annotation();

    REQUIRE(input_json.dump(2) == expected_output.dump(2));
  }

  SECTION("Get type from built-in function call")
  {
    std::istringstream input_data(R"json({
    "_type": "Module",
    "body": [
        {
            "_type": "Assign",
            "col_offset": 0,
            "end_col_offset": 10,
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
                    "id": "x",
                    "lineno": 1
                }
            ],
            "type_comment": null,
            "value": {
                "_type": "Call",
                "args": [
                    {
                        "_type": "Constant",
                        "col_offset": 8,
                        "end_col_offset": 9,
                        "end_lineno": 1,
                        "kind": null,
                        "lineno": 1,
                        "n": 1,
                        "s": 1,
                        "value": 1
                    }
                ],
                "col_offset": 4,
                "end_col_offset": 10,
                "end_lineno": 1,
                "func": {
                    "_type": "Name",
                    "col_offset": 4,
                    "ctx": {
                        "_type": "Load"
                    },
                    "end_col_offset": 7,
                    "end_lineno": 1,
                    "id": "int",
                    "lineno": 1
                },
                "keywords": [],
                "lineno": 1
            }
        }
    ],
    "filename": "blah.py",
    "type_ignores": []
    })json");

    nlohmann::json input_json;
    input_data >> input_json;

    std::istringstream output_data(R"json({
    "_type": "Module",
    "body": [
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
            "end_col_offset": 14,
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
                "id": "x",
                "lineno": 1
            },
            "value": {
                "_type": "Call",
                "args": [
                    {
                        "_type": "Constant",
                        "col_offset": 12,
                        "end_col_offset": 13,
                        "end_lineno": 1,
                        "kind": null,
                        "lineno": 1,
                        "n": 1,
                        "s": 1,
                        "value": 1
                    }
                ],
                "col_offset": 8,
                "end_col_offset": 14,
                "end_lineno": 1,
                "func": {
                    "_type": "Name",
                    "col_offset": 8,
                    "ctx": {
                        "_type": "Load"
                    },
                    "end_col_offset": 11,
                    "end_lineno": 1,
                    "id": "int",
                    "lineno": 1
                },
                "keywords": [],
                "lineno": 1
            }
        }
    ],
    "filename": "blah.py",
    "type_ignores": []
    })json");

    nlohmann::json expected_output;
    output_data >> expected_output;

    global_scope gs;
    python_annotation<nlohmann::json> ann(input_json, gs);
    ann.add_type_annotation();

    REQUIRE(input_json == expected_output);
  }
}
