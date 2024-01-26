/*******************************************************************
 Module: Filesystem fuzz tests

 Author: Rafael Sá Menezes

 Date: October 2021

 Fuzz Plan:
   - get_unique_tmp_folder
 \*******************************************************************/

/**
 * The idea is to initialize a random number of files (up to 50) and see
 *  if they are unique
 *
 */

#include <util/filesystem.h>
#include <set>
#include <string>
#include <assert.h>
void test_tmp_folder(const unsigned int *Data, size_t Size)
{
  if (Size != 1)
    return;

  const unsigned int MAX_LENGTH = 10;
  const char *format = "esbmc-fuzz-%%%%-%%%%-%%%%";

  unsigned test_length = Data[0] % MAX_LENGTH;
  if (test_length > MAX_LENGTH) // to prevent some cast issue
    return;
  std::set<std::string> names;

  for (int i = 0; i < test_length; i++)
  {
    auto entry = file_operations::get_unique_tmp_path(format);
    assert(
      !names.count(entry)); // there should be 0 elements in the set with entry
    names.insert(entry);
  }

  assert(names.size() == test_length);
}

extern "C" int LLVMFuzzerTestOneInput(const unsigned int *Data, size_t Size)
{
  test_tmp_folder(Data, Size);
  return 0;
}
