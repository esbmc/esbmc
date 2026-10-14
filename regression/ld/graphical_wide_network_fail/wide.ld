<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="regression" productVersion="1" creationDateTime="2026-07-30T00:00:00"/>
  <contentHeader name="path_explosion">
    <coordinateInfo><ld><scaling x="10" y="10"/></ld></coordinateInfo>
  </contentHeader>
  <types>
    <dataTypes/>
    <pous>
      <pou name="wide_net" pouType="program">
        <interface>
          <localVars>
            <variable name="c0_0" address="%IX0.0"><type><BOOL/></type></variable>
            <variable name="c0_1" address="%IX0.1"><type><BOOL/></type></variable>
            <variable name="c1_0" address="%IX1.0"><type><BOOL/></type></variable>
            <variable name="c1_1" address="%IX1.1"><type><BOOL/></type></variable>
            <variable name="c2_0" address="%IX2.0"><type><BOOL/></type></variable>
            <variable name="c2_1" address="%IX2.1"><type><BOOL/></type></variable>
            <variable name="c3_0" address="%IX3.0"><type><BOOL/></type></variable>
            <variable name="c3_1" address="%IX3.1"><type><BOOL/></type></variable>
            <variable name="c4_0" address="%IX4.0"><type><BOOL/></type></variable>
            <variable name="c4_1" address="%IX4.1"><type><BOOL/></type></variable>
            <variable name="c5_0" address="%IX5.0"><type><BOOL/></type></variable>
            <variable name="c5_1" address="%IX5.1"><type><BOOL/></type></variable>
            <variable name="c6_0" address="%IX6.0"><type><BOOL/></type></variable>
            <variable name="c6_1" address="%IX6.1"><type><BOOL/></type></variable>
            <variable name="c7_0" address="%IX7.0"><type><BOOL/></type></variable>
            <variable name="c7_1" address="%IX7.1"><type><BOOL/></type></variable>
            <variable name="c8_0" address="%IX8.0"><type><BOOL/></type></variable>
            <variable name="c8_1" address="%IX8.1"><type><BOOL/></type></variable>
            <variable name="c9_0" address="%IX9.0"><type><BOOL/></type></variable>
            <variable name="c9_1" address="%IX9.1"><type><BOOL/></type></variable>
            <variable name="c10_0" address="%IX10.0"><type><BOOL/></type></variable>
            <variable name="c10_1" address="%IX10.1"><type><BOOL/></type></variable>
            <variable name="c11_0" address="%IX11.0"><type><BOOL/></type></variable>
            <variable name="c11_1" address="%IX11.1"><type><BOOL/></type></variable>
            <variable name="c12_0" address="%IX12.0"><type><BOOL/></type></variable>
            <variable name="c12_1" address="%IX12.1"><type><BOOL/></type></variable>
            <variable name="c13_0" address="%IX13.0"><type><BOOL/></type></variable>
            <variable name="c13_1" address="%IX13.1"><type><BOOL/></type></variable>
            <variable name="c14_0" address="%IX14.0"><type><BOOL/></type></variable>
            <variable name="c14_1" address="%IX14.1"><type><BOOL/></type></variable>
            <variable name="c15_0" address="%IX15.0"><type><BOOL/></type></variable>
            <variable name="c15_1" address="%IX15.1"><type><BOOL/></type></variable>
            <variable name="c16_0" address="%IX16.0"><type><BOOL/></type></variable>
            <variable name="c16_1" address="%IX16.1"><type><BOOL/></type></variable>
            <variable name="c17_0" address="%IX17.0"><type><BOOL/></type></variable>
            <variable name="c17_1" address="%IX17.1"><type><BOOL/></type></variable>
            <variable name="out" address="%QX0.0"><type><BOOL/></type></variable>
          </localVars>
        </interface>
        <body>
          <LD>
            <leftPowerRail localId="1" width="10" height="40">
              <position x="30" y="20"/>
              <connectionPointOut><relPosition x="10" y="20"/></connectionPointOut>
            </leftPowerRail>
            <contact localId="10" width="20" height="20" negated="false">
              <position x="60" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c0_0</variable>
            </contact>
            <contact localId="11" width="20" height="20" negated="false">
              <position x="60" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c0_1</variable>
            </contact>
            <contact localId="12" width="20" height="20" negated="false">
              <position x="100" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="10"/>
                <connection refLocalId="11"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c1_0</variable>
            </contact>
            <contact localId="13" width="20" height="20" negated="false">
              <position x="100" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="10"/>
                <connection refLocalId="11"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c1_1</variable>
            </contact>
            <contact localId="14" width="20" height="20" negated="false">
              <position x="140" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="12"/>
                <connection refLocalId="13"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c2_0</variable>
            </contact>
            <contact localId="15" width="20" height="20" negated="false">
              <position x="140" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="12"/>
                <connection refLocalId="13"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c2_1</variable>
            </contact>
            <contact localId="16" width="20" height="20" negated="false">
              <position x="180" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="14"/>
                <connection refLocalId="15"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c3_0</variable>
            </contact>
            <contact localId="17" width="20" height="20" negated="false">
              <position x="180" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="14"/>
                <connection refLocalId="15"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c3_1</variable>
            </contact>
            <contact localId="18" width="20" height="20" negated="false">
              <position x="220" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="16"/>
                <connection refLocalId="17"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c4_0</variable>
            </contact>
            <contact localId="19" width="20" height="20" negated="false">
              <position x="220" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="16"/>
                <connection refLocalId="17"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c4_1</variable>
            </contact>
            <contact localId="20" width="20" height="20" negated="false">
              <position x="260" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="18"/>
                <connection refLocalId="19"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c5_0</variable>
            </contact>
            <contact localId="21" width="20" height="20" negated="false">
              <position x="260" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="18"/>
                <connection refLocalId="19"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c5_1</variable>
            </contact>
            <contact localId="22" width="20" height="20" negated="false">
              <position x="300" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="20"/>
                <connection refLocalId="21"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c6_0</variable>
            </contact>
            <contact localId="23" width="20" height="20" negated="false">
              <position x="300" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="20"/>
                <connection refLocalId="21"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c6_1</variable>
            </contact>
            <contact localId="24" width="20" height="20" negated="false">
              <position x="340" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="22"/>
                <connection refLocalId="23"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c7_0</variable>
            </contact>
            <contact localId="25" width="20" height="20" negated="false">
              <position x="340" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="22"/>
                <connection refLocalId="23"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c7_1</variable>
            </contact>
            <contact localId="26" width="20" height="20" negated="false">
              <position x="380" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="24"/>
                <connection refLocalId="25"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c8_0</variable>
            </contact>
            <contact localId="27" width="20" height="20" negated="false">
              <position x="380" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="24"/>
                <connection refLocalId="25"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c8_1</variable>
            </contact>
            <contact localId="28" width="20" height="20" negated="false">
              <position x="420" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="26"/>
                <connection refLocalId="27"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c9_0</variable>
            </contact>
            <contact localId="29" width="20" height="20" negated="false">
              <position x="420" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="26"/>
                <connection refLocalId="27"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c9_1</variable>
            </contact>
            <contact localId="30" width="20" height="20" negated="false">
              <position x="460" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="28"/>
                <connection refLocalId="29"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c10_0</variable>
            </contact>
            <contact localId="31" width="20" height="20" negated="false">
              <position x="460" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="28"/>
                <connection refLocalId="29"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c10_1</variable>
            </contact>
            <contact localId="32" width="20" height="20" negated="false">
              <position x="500" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="30"/>
                <connection refLocalId="31"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c11_0</variable>
            </contact>
            <contact localId="33" width="20" height="20" negated="false">
              <position x="500" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="30"/>
                <connection refLocalId="31"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c11_1</variable>
            </contact>
            <contact localId="34" width="20" height="20" negated="false">
              <position x="540" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="32"/>
                <connection refLocalId="33"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c12_0</variable>
            </contact>
            <contact localId="35" width="20" height="20" negated="false">
              <position x="540" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="32"/>
                <connection refLocalId="33"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c12_1</variable>
            </contact>
            <contact localId="36" width="20" height="20" negated="false">
              <position x="580" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="34"/>
                <connection refLocalId="35"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c13_0</variable>
            </contact>
            <contact localId="37" width="20" height="20" negated="false">
              <position x="580" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="34"/>
                <connection refLocalId="35"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c13_1</variable>
            </contact>
            <contact localId="38" width="20" height="20" negated="false">
              <position x="620" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="36"/>
                <connection refLocalId="37"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c14_0</variable>
            </contact>
            <contact localId="39" width="20" height="20" negated="false">
              <position x="620" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="36"/>
                <connection refLocalId="37"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c14_1</variable>
            </contact>
            <contact localId="40" width="20" height="20" negated="false">
              <position x="660" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="38"/>
                <connection refLocalId="39"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c15_0</variable>
            </contact>
            <contact localId="41" width="20" height="20" negated="false">
              <position x="660" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="38"/>
                <connection refLocalId="39"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c15_1</variable>
            </contact>
            <contact localId="42" width="20" height="20" negated="false">
              <position x="700" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="40"/>
                <connection refLocalId="41"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c16_0</variable>
            </contact>
            <contact localId="43" width="20" height="20" negated="false">
              <position x="700" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="40"/>
                <connection refLocalId="41"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c16_1</variable>
            </contact>
            <contact localId="44" width="20" height="20" negated="false">
              <position x="740" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="42"/>
                <connection refLocalId="43"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c17_0</variable>
            </contact>
            <contact localId="45" width="20" height="20" negated="false">
              <position x="740" y="40"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="42"/>
                <connection refLocalId="43"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>c17_1</variable>
            </contact>
            <coil localId="46" width="20" height="20" negated="false">
              <position x="780" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="44"/>
                <connection refLocalId="45"/>
              </connectionPointIn>
              <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
              <variable>out</variable>
            </coil>
            <rightPowerRail localId="2" width="10" height="40">
              <position x="900" y="20"/>
              <connectionPointIn>
                <relPosition x="0" y="20"/>
                <connection refLocalId="46"/>
              </connectionPointIn>
            </rightPowerRail>
          </LD>
        </body>
      </pou>
    </pous>
  </types>
  <instances><configurations><configuration name="config"><resource name="res">
    <task name="main" interval="T#10ms" priority="0"><pouInstance name="inst" typeName="wide_net"/></task>
  </resource></configuration></configurations></instances>
</project>
