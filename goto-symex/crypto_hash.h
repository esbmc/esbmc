#ifndef _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_
#define _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_

#ifndef NO_OPENSSL

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <openssl/sha.h>
}

#endif /* NO_OPENSSL */

#include <string>

#define CRYPTO_HASH_SIZE	32

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

#ifndef NO_OPENSSL
  static bool have_pointers;
  static int (*sha_init)(SHA256_CTX *c);
  static int (*sha_update)(SHA256_CTX *c, const void *data, size_t len);
  static int (*sha_final)(unsigned char *md, SHA256_CTX *c);
  static void setup_pointers();
#endif
};

#endif /* _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_ */
