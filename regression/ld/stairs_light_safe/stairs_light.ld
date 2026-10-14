<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xhtml="http://www.w3.org/1999/xhtml" xmlns:ns1="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="Desconocido" productName="Sin nombre" productVersion="1" creationDateTime="2024-11-18T12:32:42"/>
  <contentHeader name="Sin nombre" modificationDateTime="2024-12-05T12:24:50">
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
      <pou name="light_control" pouType="program">
        <interface>
          <localVars>
            <variable name="stairs_light" address="%QX0.0">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Light switch turns on with true value]]></xhtml:p>
              </documentation>
            </variable>
          </localVars>
          <localVars>
            <variable name="lights_buttons_state">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Variable that indicates the state of lights according the buttons actions]]></xhtml:p>
              </documentation>
            </variable>
          </localVars>
          <localVars>
            <variable name="stairs_pir_sensor" address="%IX0.0">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Passive Infrared Sensor detects persons arriving the stairs]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="control_button_down" address="%IX0.1">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Press button located down the stairs]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="control_button_up" address="%IX0.2">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Press button located up the stairs]]></xhtml:p>
              </documentation>
            </variable>
          </localVars>
          <localVars>
            <variable name="TOF0">
              <type>
                <derived name="TOF"/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Temporizador que marca el tiempo de encendido de la luz]]></xhtml:p>
              </documentation>
            </variable>
          </localVars>
        </interface>
        <body>
          <LD>
            <leftPowerRail localId="1" width="10" height="280">
              <position x="30" y="200"/>
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
            </leftPowerRail>
            <rightPowerRail localId="2" width="10" height="280">
              <position x="710" y="200"/>
              <connectionPointIn>
                <relPosition x="0" y="20"/>
                <connection refLocalId="5">
                  <position x="710" y="220"/>
                  <position x="640" y="220"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="60"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="100"/>
                <connection refLocalId="6">
                  <position x="710" y="300"/>
                  <position x="640" y="300"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="140"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="180"/>
                <connection refLocalId="11">
                  <position x="710" y="380"/>
                  <position x="640" y="380"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="220"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="260"/>
              </connectionPointIn>
            </rightPowerRail>
            <contact localId="3" negated="false" edge="rising" width="30" height="20">
              <position x="110" y="210"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="110" y="220"/>
                  <position x="40" y="220"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>control_button_up</variable>
            </contact>
            <contact localId="4" negated="false" edge="rising" width="30" height="20">
              <position x="110" y="290"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="110" y="300"/>
                  <position x="40" y="300"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>control_button_down</variable>
            </contact>
            <coil localId="5" negated="false" width="30" height="20" storage="set">
              <position x="610" y="210"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="8">
                  <position x="610" y="220"/>
                  <position x="530" y="220"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>lights_buttons_state</variable>
            </coil>
            <coil localId="6" negated="false" width="30" height="20" storage="reset">
              <position x="610" y="290"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="7">
                  <position x="610" y="300"/>
                  <position x="530" y="300"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>lights_buttons_state</variable>
            </coil>
            <contact localId="7" negated="false" width="30" height="20">
              <position x="500" y="290"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="3">
                  <position x="500" y="300"/>
                  <position x="335" y="300"/>
                  <position x="335" y="220"/>
                  <position x="140" y="220"/>
                </connection>
                <connection refLocalId="4">
                  <position x="500" y="300"/>
                  <position x="140" y="300"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>lights_buttons_state</variable>
            </contact>
            <contact localId="8" negated="true" width="30" height="20">
              <position x="500" y="210"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="4">
                  <position x="500" y="220"/>
                  <position x="335" y="220"/>
                  <position x="335" y="300"/>
                  <position x="140" y="300"/>
                </connection>
                <connection refLocalId="3">
                  <position x="500" y="220"/>
                  <position x="140" y="220"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>lights_buttons_state</variable>
            </contact>
            <contact localId="9" negated="false" edge="rising" width="30" height="20">
              <position x="110" y="370"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="110" y="380"/>
                  <position x="40" y="380"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>stairs_pir_sensor</variable>
            </contact>
            <block localId="10" typeName="TOF" instanceName="TOF0" width="50" height="60">
              <position x="330" y="350"/>
              <inputVariables>
                <variable formalParameter="IN">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="12">
                      <position x="330" y="380"/>
                      <position x="250" y="380"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="PT">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="14">
                      <position x="330" y="400"/>
                      <position x="310" y="400"/>
                    </connection>
                  </connectionPointIn>
                </variable>
              </inputVariables>
              <inOutVariables/>
              <outputVariables>
                <variable formalParameter="Q">
                  <connectionPointOut>
                    <relPosition x="50" y="30"/>
                  </connectionPointOut>
                </variable>
                <variable formalParameter="ET">
                  <connectionPointOut>
                    <relPosition x="50" y="50"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
            <coil localId="11" negated="false" width="30" height="20">
              <position x="610" y="370"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="10" formalParameter="Q">
                  <position x="610" y="380"/>
                  <position x="380" y="380"/>
                </connection>
                <connection refLocalId="13">
                  <position x="610" y="380"/>
                  <position x="550" y="380"/>
                  <position x="550" y="460"/>
                  <position x="140" y="460"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>stairs_light</variable>
            </coil>
            <contact localId="12" negated="true" width="30" height="20">
              <position x="220" y="370"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="9">
                  <position x="220" y="380"/>
                  <position x="140" y="380"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>lights_buttons_state</variable>
            </contact>
            <contact localId="13" negated="false" width="30" height="20">
              <position x="110" y="450"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="110" y="460"/>
                  <position x="40" y="460"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>lights_buttons_state</variable>
            </contact>
            <inVariable localId="14" width="40" height="20" negated="false">
              <position x="270" y="390"/>
              <connectionPointOut>
                <relPosition x="40" y="10"/>
              </connectionPointOut>
              <expression>T#20s</expression>
            </inVariable>
            <comment localId="15" height="160" width="1170">
              <position x="10" y="10"/>
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
            <comment localId="16" height="60" width="320">
              <position x="730" y="230"/>
              <content>
                <xhtml:p><![CDATA[When any button is pressed and is detected a rise pulse the light state flag change among true and false, on and off]]></xhtml:p>
              </content>
            </comment>
            <comment localId="17" height="70" width="310">
              <position x="730" y="380"/>
              <content>
                <xhtml:p><![CDATA[If sensor PIR signal is detected and the light is turned off, light outuput will turn on during preset time, 20 seconds, if state flag is on the light putput will turn on at the same time]]></xhtml:p>
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
            <pouInstance name="instance0" typeName="light_control"/>
          </task>
        </resource>
      </configuration>
    </configurations>
  </instances>
</project>
