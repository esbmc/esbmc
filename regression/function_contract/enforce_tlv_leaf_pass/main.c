/*
 * TLV (Tag-Length-Value) message parser with compositional contracts.
 *
 * Three-level call hierarchy:
 *   parse_message -> parse_header -> validate_tag
 *
 * Tests: enforce each function's contract, then replace "*" to verify
 * the system property in main using only contracts.
 *
 * The TLV format:
 *   byte 0:    tag   (must be 0x01..0x04)
 *   byte 1:    length of value field (0..125, short form)
 *   byte 2..:  value payload
 *
 * parse_message returns the total consumed bytes (hdr + payload)
 * through *consumed, or -1 on error.
 */

typedef unsigned char u8;
typedef unsigned int  u32;

/* ---------- Level 3 (leaf): validate_tag ---------- */

int validate_tag(u8 tag)
{
    __ESBMC_ensures(
        (__ESBMC_return_value == 0) | (__ESBMC_return_value == -1)
    );
    /* tag must be in the range 0x01 .. 0x04 */
    __ESBMC_ensures(
        (__ESBMC_return_value != 0) | (tag >= 0x01 & tag <= 0x04)
    );
    __ESBMC_ensures(
        (__ESBMC_return_value == 0) | (tag < 0x01 | tag > 0x04)
    );

    if (tag >= 0x01 && tag <= 0x04)
        return 0;
    return -1;
}

/* ---------- Level 2 (mid): parse_header ---------- */

int parse_header(const u8 *buf, u32 buf_len,
                 u32 *hdr_len, u32 *payload_len)
{
    __ESBMC_requires(buf != ((void *)0));
    __ESBMC_requires(hdr_len != ((void *)0));
    __ESBMC_requires(payload_len != ((void *)0));
    __ESBMC_requires(buf_len > 0);

    __ESBMC_assigns(*hdr_len, *payload_len);

    /* return is 0 (ok) or -1 (error) */
    __ESBMC_ensures(
        (__ESBMC_return_value == 0) | (__ESBMC_return_value == -1)
    );
    /* on success: header is 2 bytes, payload_len <= 125,
       and the total fits inside buf_len */
    __ESBMC_ensures(
        (__ESBMC_return_value != 0) | (*hdr_len == 2)
    );
    __ESBMC_ensures(
        (__ESBMC_return_value != 0) | (*payload_len <= 125)
    );
    __ESBMC_ensures(
        (__ESBMC_return_value != 0) | (*hdr_len + *payload_len <= buf_len)
    );
    /* consumed > 0 on success */
    __ESBMC_ensures(
        (__ESBMC_return_value != 0) | (*hdr_len + *payload_len >= 2)
    );

    if (buf_len < 2)
        return -1;

    /* validate tag via leaf function */
    int ret = validate_tag(buf[0]);
    if (ret)
        return -1;

    /* short-form length */
    u8 len_byte = buf[1];
    if (len_byte > 125)
        return -1;

    *hdr_len     = 2;
    *payload_len = len_byte;

    if (*hdr_len + *payload_len > buf_len)
        return -1;

    return 0;
}

/* ---------- Level 1 (top): parse_message ---------- */

int parse_message(const u8 *buf, u32 buf_len, u32 *consumed)
{
    __ESBMC_requires(consumed != ((void *)0));
    __ESBMC_requires(!(buf_len > 0) || (buf != ((void *)0)));
    __ESBMC_requires(buf_len <= 128);

    __ESBMC_assigns(*consumed);

    /* return is 0 (ok) or -1 (error) */
    __ESBMC_ensures(
        (__ESBMC_return_value == 0) | (__ESBMC_return_value == -1)
    );
    /* on success: consumed > 0 and consumed <= buf_len */
    __ESBMC_ensures(
        (__ESBMC_return_value != 0) | (*consumed > 0)
    );
    __ESBMC_ensures(
        (__ESBMC_return_value != 0) | (*consumed <= buf_len)
    );

    if (buf == ((void *)0) || buf_len == 0 || consumed == ((void *)0))
        return -1;

    u32 hdr_len = 0, payload_len = 0;

    int ret = parse_header(buf, buf_len, &hdr_len, &payload_len);
    if (ret)
        return -1;

    *consumed = hdr_len + payload_len;
    return 0;
}

/* ---------- main ---------- */

int main(void)
{
    u8  buffer[128];
    u32 len;
    u32 consumed;

    __ESBMC_assume(len > 0 && len <= 128);

    int ret = parse_message(buffer, len, &consumed);

    /* system property: if parsing succeeds, consumed is sane */
    if (ret == 0) {
        assert(consumed > 0);
        assert(consumed <= len);
    }

    return 0;
}
