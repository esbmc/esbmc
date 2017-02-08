#ifndef _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_
#define _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_

#include <ac_config.h>

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef HAVE_OPENSSL
#include <openssl/sha.h>
#endif /* NO_OPENSSL */
}


#include <string>

#define CRYPTO_HASH_SIZE	32

class crypto_hash {
  public:
  uint8_t hash[32];
#ifndef HAVE_OPENSSL
  SHA_CTX c;
#endif

  bool operator<(const crypto_hash h2) const;

  std::string to_string() const;

  crypto_hash();
  void ingest(void const *data, unsigned int size);
  void fin();
};

#endif /* _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_ */
