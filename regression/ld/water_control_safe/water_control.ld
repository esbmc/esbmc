<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xhtml="http://www.w3.org/1999/xhtml" xmlns:ns1="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="Desconocido" productName="Sin nombre" productVersion="1" creationDateTime="2024-11-13T10:20:43"/>
  <contentHeader name="Water_Reserve_Control" modificationDateTime="2024-12-05T12:40:23">
    <coordinateInfo>
      <fbd>
        <scaling x="10" y="10"/>
      </fbd>
      <ld>
        <scaling x="10" y="10"/>
      </ld>
      <sfc>
        <scaling x="10" y="10"/>
      </sfc>
    </coordinateInfo>
  </contentHeader>
  <types>
    <dataTypes/>
    <pous>
      <pou name="Water_Control" pouType="program">
        <interface>
          <localVars>
            <variable name="Pool_Low_Level_Sensor" address="%IX0.0">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Pools Level Sensor signal of min level. Zero indicates that level is under the minimun]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Tank_High_Level_Sensor" address="%IX0.1">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Tank Level Sensor of miax level. State "1" indicates maximun level]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Water_Pump" address="%QX0.0">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Pump state control output. High value turns on the motor]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Tank_Low_Level_Sensor" address="%IX0.2">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Tank Level Sensor of min level. Zero indicates that level is under the minimun]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Automatic_Manual_Switch" address="%IX0.3">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Automatic/Manual selector. State "0" is Manual mode and "1" Automatic mode]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Stop_Button" address="%IX0.4">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Stop Push Button. If changes its value to "1" indicates stop de pump  ]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Start_Button" address="%IX0.5">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Start Push Button. If changes its value to "1" indicates start de pump  ]]></xhtml:p>
              </documentation>
            </variable>
          </localVars>
        </interface>
        <body>
          <LD>
            <leftPowerRail localId="1" width="10" height="360">
              <position x="20" y="180"/>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="20"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="60"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="100"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="140"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="180"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="220"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="260"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="300"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="340"/>
              </connectionPointOut>
            </leftPowerRail>
            <rightPowerRail localId="2" width="10" height="360">
              <position x="690" y="180"/>
              <connectionPointIn>
                <relPosition x="0" y="20"/>
                <connection refLocalId="4">
                  <position x="690" y="200"/>
                  <position x="640" y="200"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="60"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="100"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="140"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="180"/>
                <connection refLocalId="8">
                  <position x="690" y="360"/>
                  <position x="640" y="360"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="220"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="260"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="300"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="340"/>
              </connectionPointIn>
            </rightPowerRail>
            <contact localId="3" negated="false" width="30" height="20">
              <position x="230" y="190"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="9">
                  <position x="230" y="200"/>
                  <position x="130" y="200"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Pool_Low_Level_Sensor</variable>
            </contact>
            <coil localId="4" negated="false" width="30" height="20" storage="set">
              <position x="610" y="190"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="6">
                  <position x="610" y="200"/>
                  <position x="500" y="200"/>
                </connection>
                <connection refLocalId="12">
                  <position x="610" y="200"/>
                  <position x="560" y="200"/>
                  <position x="560" y="280"/>
                  <position x="380" y="280"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Water_Pump</variable>
            </coil>
            <contact localId="5" negated="true" width="30" height="20">
              <position x="350" y="190"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="3">
                  <position x="350" y="200"/>
                  <position x="260" y="200"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Tank_Low_Level_Sensor</variable>
            </contact>
            <contact localId="6" negated="true" width="30" height="20">
              <position x="470" y="190"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="5">
                  <position x="470" y="200"/>
                  <position x="380" y="200"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Tank_High_Level_Sensor</variable>
            </contact>
            <contact localId="7" negated="true" width="30" height="20" executionOrderId="0">
              <position x="100" y="350"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="100" y="360"/>
                  <position x="30" y="360"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Pool_Low_Level_Sensor</variable>
            </contact>
            <coil localId="8" negated="false" width="30" height="20" storage="reset" executionOrderId="0">
              <position x="610" y="350"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="7">
                  <position x="610" y="360"/>
                  <position x="130" y="360"/>
                </connection>
                <connection refLocalId="13">
                  <position x="610" y="360"/>
                  <position x="560" y="360"/>
                  <position x="560" y="520"/>
                  <position x="130" y="520"/>
                </connection>
                <connection refLocalId="14">
                  <position x="610" y="360"/>
                  <position x="560" y="360"/>
                  <position x="560" y="440"/>
                  <position x="130" y="440"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Water_Pump</variable>
            </coil>
            <contact localId="9" negated="false" width="30" height="20">
              <position x="100" y="190"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="100" y="200"/>
                  <position x="30" y="200"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Automatic_Manual_Switch</variable>
            </contact>
            <contact localId="10" negated="false" width="30" height="20" executionOrderId="0">
              <position x="100" y="270"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="100" y="280"/>
                  <position x="30" y="280"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Start_Button</variable>
            </contact>
            <contact localId="11" negated="false" width="30" height="20" executionOrderId="0">
              <position x="230" y="270"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="10">
                  <position x="230" y="280"/>
                  <position x="130" y="280"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Pool_Low_Level_Sensor</variable>
            </contact>
            <contact localId="12" negated="true" width="30" height="20" executionOrderId="0">
              <position x="350" y="270"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="11">
                  <position x="350" y="280"/>
                  <position x="260" y="280"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Tank_High_Level_Sensor</variable>
            </contact>
            <contact localId="13" negated="false" width="30" height="20" executionOrderId="0">
              <position x="100" y="510"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="100" y="520"/>
                  <position x="30" y="520"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Stop_Button</variable>
            </contact>
            <contact localId="14" negated="false" width="30" height="20" executionOrderId="0">
              <position x="100" y="430"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="100" y="440"/>
                  <position x="30" y="440"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Tank_High_Level_Sensor</variable>
            </contact>
            <comment localId="15" height="160" width="1170">
              <position x="0" y="0"/>
              <content>
                <xhtml:p><![CDATA[OpenPLC HAL for CONTROLLINO MAXI Automation see https://www.controllino.com/wp-content/uploads/2021/03/CONTROLLINO-MAXI-Automation-Pinout.pdf

Digital In:  AI2, AI3, AI4, AI5, AI6, AI7, AI8, AI9     (%IX0.0 - %IX0.7)
             AI10, AI11, DI0, DI1, DI2, DI3, AI2, AI3   (%IX1.0 - %IX1.7)

Digital Out: DO0, DO1, DO2, DO3, DO4, DO5, DO6, DO7     (%QX0.0 - %QX0.7)
             R0, R1, R2, R3, R4, R5, R6, R7             (%QX1.0 - %QX1.7)
             R8, R9                                     (%QX2.0 - %QX2.1)

Analog In:   AI0, AI1, AI13, AI13                       (%IW0 - %IW3)

Analog Out:  AO0, AO1                                   (%QW0 - %QW1)
]]></xhtml:p>
              </content>
            </comment>
            <comment localId="16" height="110" width="340">
              <position x="720" y="190"/>
              <content>
                <xhtml:p><![CDATA[Behaviour: The pump starts if the tank level is under de low limit in this case both level sensors into the tank will be zero, negative coil value. The second condition to start de pump is that level of the pool be over the minimum level and the switch be changed to automatic “true”.  The oter condition is that start button be pressed and the level conditions described be acomplished]]></xhtml:p>
              </content>
            </comment>
            <comment localId="17" height="80" width="310">
              <position x="720" y="360"/>
              <content>
                <xhtml:p><![CDATA[If pump is working and any level condition change to wrong state or stop button is pressed then pump turns off]]></xhtml:p>
              </content>
            </comment>
          </LD>
        </body>
      </pou>
    </pous>
  </types>
  <instances>
    <configurations>
      <configuration name="Config0">
        <resource name="Res0">
          <task name="task0" priority="0" interval="T#20ms">
            <pouInstance name="instance0" typeName="Water_Control"/>
          </task>
        </resource>
      </configuration>
    </configurations>
  </instances>
</project>
