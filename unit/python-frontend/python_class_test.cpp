#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <python-frontend/python_class.h>

using json = nlohmann::json;

//
// Corresponding Python code:
//
// class Animal:
//     kingdom = "Animalia"
//
//     def __init__(self, name):
//         self.name = name
//
//     def speak(self):
//         pass
//

TEST_CASE("python_class::parse - Simple ClassDef (Animal)")
{
  // Passing only the ClassDef node
  std::istringstream data(R"json(
  {
    "_type": "ClassDef",
    "name": "Animal",
    "bases": [],
    "body": [
      {
        "_type": "Assign",
        "targets": [ { "_type": "Name", "id": "kingdom", "ctx": {"_type":"Store"} } ],
        "value":   { "_type": "Constant", "value": "Animalia" }
      },
      {
        "_type": "FunctionDef",
        "name": "__init__",
        "args": { "_type":"arguments", "args": [], "defaults": [], "kw_defaults": [], "kwonlyargs": [], "posonlyargs": [], "kwarg": null, "vararg": null },
        "body": [
          {
            "_type": "Assign",
            "targets": [
              { "_type":"Attribute", "attr":"name", "value":{"_type":"Name","id":"self","ctx":{"_type":"Load"}}, "ctx":{"_type":"Store"} }
            ],
            "value": { "_type":"Name","id":"name","ctx":{"_type":"Load"} }
          }
        ],
        "decorator_list": []
      },
      {
        "_type": "FunctionDef",
        "name": "speak",
        "args": { "_type":"arguments", "args": [], "defaults": [], "kw_defaults": [], "kwonlyargs": [], "posonlyargs": [], "kwarg": null, "vararg": null },
        "body": [],
        "decorator_list": []
      }
    ]
  })json");

  json cls_json;
  data >> cls_json;

  python_class pc;
  pc.parse(cls_json);

  REQUIRE(pc.name() == "Animal");
  REQUIRE(pc.methods().count("__init__") == 1);
  REQUIRE(pc.methods().count("speak") == 1);

  // Should capture only class-level attributes (Assign -> Name)
  // and ignore instance attributes defined inside __init__
  REQUIRE(pc.attributes().count("kingdom") == 1);
  REQUIRE(pc.attributes().count("name") == 0);

  REQUIRE(pc.bases().empty());
}

//
// Corresponding Python code:
//
// class Dog(Animal):
//     def __init__(self):
//         pass
//
//     def speak(self):
//         pass
//

TEST_CASE("python_class::parse - ClassDef with a simple base (Dog : Animal)")
{
  std::istringstream data(R"json(
  {
    "_type": "ClassDef",
    "name": "Dog",
    "bases": [
      { "_type": "Name", "id": "Animal", "ctx": { "_type": "Load" } }
    ],
    "body": [
      { "_type": "FunctionDef", "name": "__init__", "args": { "_type":"arguments", "args": [], "defaults": [], "kw_defaults": [], "kwonlyargs": [], "posonlyargs": [], "kwarg": null, "vararg": null }, "body": [], "decorator_list": [] },
      { "_type": "FunctionDef", "name": "speak",    "args": { "_type":"arguments", "args": [], "defaults": [], "kw_defaults": [], "kwonlyargs": [], "posonlyargs": [], "kwarg": null, "vararg": null }, "body": [], "decorator_list": [] }
    ]
  })json");

  json cls_json;
  data >> cls_json;

  python_class pc;
  pc.parse(cls_json);

  REQUIRE(pc.name() == "Dog");
  REQUIRE(pc.methods().count("__init__") == 1);
  REQUIRE(pc.methods().count("speak") == 1);
  REQUIRE(pc.attributes().empty());

  // Should collect Animal as the only base
  REQUIRE(pc.bases().count("Animal") == 1);
}

//
// Corresponding Python code:
//
// class MyClass(pkg.sub.Base):
//     def __init__(self):
//         pass
//

TEST_CASE("python_class::parse - Dotted base name (MyClass : pkg.sub.Base)")
{
  std::istringstream data(R"json(
  {
    "_type": "ClassDef",
    "name": "MyClass",
    "bases": [
      {
        "_type": "Attribute",
        "attr": "Base",
        "value": {
          "_type": "Attribute",
          "attr": "sub",
          "value": { "_type": "Name", "id": "pkg" }
        }
      }
    ],
    "body": [
      { "_type": "FunctionDef", "name": "__init__", "args": { "_type":"arguments", "args": [], "defaults": [], "kw_defaults": [], "kwonlyargs": [], "posonlyargs": [], "kwarg": null, "vararg": null }, "body": [], "decorator_list": [] }
    ]
  })json");

  json cls_json;
  data >> cls_json;

  python_class pc;
  pc.parse(cls_json);

  REQUIRE(pc.name() == "MyClass");
  REQUIRE(pc.methods().count("__init__") == 1);
  REQUIRE(pc.attributes().empty());

  // Should resolve the dotted base as "pkg.sub.Base"
  REQUIRE(pc.bases().count("pkg.sub.Base") == 1);
}
