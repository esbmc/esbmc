#include <iostream>
using namespace std;

class USBDevice
{
private:
    long m_lID;
 
public:
    USBDevice(long lID)
        : m_lID(lID)
    {
    }
 
    long GetID() { return m_lID; }
};
 
class NetworkDevice
{
private:
    long m_lID;
 
public:
    NetworkDevice(long lID)
        : m_lID(lID)
    {
    }
 
    long GetID() { return m_lID; }
};
 
class WirelessAdaptor: public USBDevice, public NetworkDevice
{
public:
    WirelessAdaptor(long lUSBID, long lNetworkID)
        : USBDevice(lUSBID), NetworkDevice(lNetworkID)
    {
    }
};
 
int main()
{
    WirelessAdaptor c54G(5442, 181742); 
    return 0;
}
