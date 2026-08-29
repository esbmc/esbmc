<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="regression" productVersion="1" creationDateTime="2026-07-30T00:00:00"/>
  <contentHeader name="contact_driven">
    <coordinateInfo>
      <ld>
        <scaling x="10" y="10"/>
      </ld>
    </coordinateInfo>
  </contentHeader>
  <types>
    <dataTypes/>
    <pous>
      <pou name="alarm_control" pouType="program">
        <interface>
          <localVars>
            <variable name="level_high" address="%IX0.0">
              <type>
                <BOOL/>
              </type>
            </variable>
            <variable name="inhibit" address="%IX0.1">
              <type>
                <BOOL/>
              </type>
            </variable>
            <variable name="alarm" address="%QX0.0">
              <type>
                <BOOL/>
              </type>
            </variable>
          </localVars>
        </interface>
        <body>
          <LD>
            <leftPowerRail localId="1" width="10" height="40">
              <position x="30" y="20"/>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="20"/>
              </connectionPointOut>
            </leftPowerRail>
            <rightPowerRail localId="2" width="10" height="40">
              <position x="410" y="20"/>
              <connectionPointIn>
                <relPosition x="0" y="20"/>
                <connection refLocalId="6"/>
              </connectionPointIn>
            </rightPowerRail>
            <contact localId="3" width="20" height="20" negated="false">
              <position x="90" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1"/>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="20" y="10"/>
              </connectionPointOut>
              <variable>level_high</variable>
            </contact>
            <contact localId="4" width="20" height="20" negated="true">
              <position x="180" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="3"/>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="20" y="10"/>
              </connectionPointOut>
              <variable>inhibit</variable>
            </contact>
            <coil localId="6" width="20" height="20" negated="false">
              <position x="330" y="10"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="4"/>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="20" y="10"/>
              </connectionPointOut>
              <variable>alarm</variable>
            </coil>
          </LD>
        </body>
      </pou>
    </pous>
  </types>
  <instances>
    <configurations>
      <configuration name="config">
        <resource name="res">
          <task name="main" interval="T#10ms" priority="0">
            <pouInstance name="inst" typeName="alarm_control"/>
          </task>
        </resource>
      </configuration>
    </configurations>
  </instances>
</project>
