// As github_4448, with the payload-size guard removed and an offset past the
// whole object: the copy really is out of bounds and must be reported (#4448).
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

extern "C" {
uint8_t nondet_u8();
size_t  nondet_size();
}

constexpr static size_t BufSize        = 512;
constexpr static size_t MaxPartSize    = 32;
constexpr static size_t MaxPayloadSize = 254;

struct [[gnu::packed]] SrcView
{
    uint8_t                              cmd;
    uint8_t                              size;
    std::array<uint8_t, MaxPartSize + 1> data;

    static SrcView& from(uint8_t* p)
    {
        return *std::bit_cast<SrcView*>(p);
    }
};

struct [[gnu::packed]] Packet
{
    std::array<uint8_t, 4>              header;
    std::array<uint8_t, MaxPayloadSize> ipmi_data;

    static Packet& from(std::array<uint8_t, BufSize>& b)
    {
        return *std::bit_cast<Packet*>(b.data());
    }
};

struct Helper {};

class Outer
{
public:
    explicit Outer(Helper& h) : _helper(h) {}
    void set_offset(size_t off) { _rx_offset = off; }

    static void
    dispatch(uint8_t* incoming, size_t& i2c_size, void* self)
    {
        static_cast<Outer*>(self)->do_write(incoming, i2c_size);
    }

private:
    void do_write(uint8_t* incoming, size_t& i2c_size)
    {
        auto& rx = SrcView::from(incoming);
        if (i2c_size == 0 || rx.size == 0 || rx.size > MaxPartSize) return;
        if (i2c_size != static_cast<size_t>(rx.size) + 2u)            return;

        auto& pkt = Packet::from(_buffer);
        switch (rx.cmd) {
            case 0x02:                       // WriteSingle
                std::copy(rx.data.begin(),
                          rx.data.begin() + rx.size,
                          pkt.ipmi_data.begin());
                _rx_size = rx.size;
                break;
            case 0x06:                       // WriteMultiStart
                if (rx.size == MaxPartSize) {
                    std::copy(rx.data.begin(),
                              rx.data.begin() + rx.size,
                              pkt.ipmi_data.begin());
                    _rx_offset = rx.size;
                }
                break;
            case 0x07:                       // WriteMultiMiddle
                if ((rx.size == MaxPartSize) && (_rx_offset > 0)
                    ) {
                    std::copy(rx.data.begin(),
                              rx.data.begin() + rx.size,
                              pkt.ipmi_data.begin() + _rx_offset);
                    _rx_offset += MaxPartSize;
                }
                break;
            default:
                break;
        }
    }

    Helper&                       _helper;
    std::array<uint8_t, BufSize>  _buffer;
    size_t                        _rx_size;
    size_t                        _rx_offset;
};

int main()
{
    static Helper helper;
    static Outer  outer{helper};

    std::array<uint8_t, 35> incoming{};
    for (size_t i = 0; i < incoming.size(); ++i) {
        incoming[i] = nondet_u8();
    }

    size_t i2c_size = nondet_size();
    __ESBMC_assume(i2c_size <= incoming.size());

    size_t off = nondet_size();
    __ESBMC_assume(off >= 560 && off <= 600);
    outer.set_offset(off);
    __ESBMC_assume(incoming[0] == 0x07 && incoming[1] == 32);
    Outer::dispatch(incoming.data(), i2c_size, &outer);
    return 0;
}
