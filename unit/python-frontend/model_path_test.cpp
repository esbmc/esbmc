#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/model_path.h>

// The parser records each module's path as the operating system spells it,
// so on Windows the model files are '\'-separated.

TEST_CASE(
  "model paths are recognised with either separator",
  "[python-frontend][model-path]")
{
  const std::string posix = "/tmp/esbmc-python-astgen-1/models/numpy.py";
  const std::string windows =
    "C:\\Users\\u\\AppData\\Local\\Temp\\esbmc-python-astgen-1\\models\\numpy."
    "py";

  CHECK(model_path::is_model(posix, "numpy.py"));
  CHECK(model_path::is_model(windows, "numpy.py"));
  CHECK(model_path::in_models(windows));
  CHECK(
    model_path::normalised(windows) ==
    "C:/Users/u/AppData/Local/Temp/esbmc-python-astgen-1/models/numpy.py");
}

TEST_CASE("user files are not models", "[python-frontend][model-path]")
{
  CHECK_FALSE(model_path::in_models("F:\\work\\main.py"));
  CHECK_FALSE(model_path::is_model("F:\\work\\numpy.py", "numpy.py"));
  CHECK_FALSE(
    model_path::is_model(
      "C:\\Temp\\astgen\\models\\numpy_extra.py", "numpy.py"));
}

TEST_CASE(
  "numpy.linalg submodules are recognised",
  "[python-frontend][model-path]")
{
  CHECK(
    model_path::is_numpy_model("C:\\Temp\\astgen\\numpy\\linalg\\__init__.py"));
  CHECK(model_path::is_numpy_model("/tmp/astgen/models/numpy.py"));
  CHECK_FALSE(model_path::is_numpy_model("F:\\work\\linalg.py"));
}
