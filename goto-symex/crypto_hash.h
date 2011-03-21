extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
}

#include <string>

class crypto_hash {
  public:
  uint8_t hash[32];
  bool valid;

  bool operator<(const crypto_hash h2) const;

  std::string to_string() const;

  crypto_hash(const uint8_t *data, int sz);
  crypto_hash(std::string str);
  crypto_hash();

  protected:
  void init(const uint8_t *data, int sz);
};
