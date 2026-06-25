<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="Desconocido" productName="Sin nombre" productVersion="1" creationDateTime="2024-11-28T09:53:41"/>
  <contentHeader name="Sin nombre" modificationDateTime="2024-12-06T04:11:27">
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
      <pou name="Dimmer" pouType="program">
        <interface>
          <localVars>
            <variable name="Control_button" address="%IX0.0">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Push button input]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Light_output" address="%QX0.0">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Light dimmer control output]]></xhtml:p>
              </documentation>
            </variable>
          </localVars>
          <localVars>
            <variable name="Light_bright">
              <type>
                <INT/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Step sequence count indicates the bright level]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Pulse_regulator">
              <type>
                <TIME/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Value in time of pulse width]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Light_on_state">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Indicates that light is on]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Reset_state">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Reset de counter, change the state to zero ]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Flag_cicle">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Flag active during a cicle, 10ms, used to restart]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="Full_bright">
              <type>
                <BOOL/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Indicates maximum bright, turn the light on directly]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="CTU0">
              <type>
                <derived name="CTU"/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[States counter]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="TP0">
              <type>
                <derived name="TP"/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Pulse generator and width corrector]]></xhtml:p>
              </documentation>
            </variable>
            <variable name="TOF0">
              <type>
                <derived name="TOF"/>
              </type>
              <documentation>
                <xhtml:p><![CDATA[Cicle generator timer]]></xhtml:p>
              </documentation>
            </variable>
          </localVars>
        </interface>
        <body>
          <LD>
            <leftPowerRail localId="1" width="10" height="800">
              <position x="100" y="150"/>
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
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="380"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="420"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="460"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="500"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="540"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="580"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="620"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="660"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="700"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="740"/>
              </connectionPointOut>
              <connectionPointOut formalParameter="">
                <relPosition x="10" y="780"/>
              </connectionPointOut>
            </leftPowerRail>
            <rightPowerRail localId="2" width="10" height="800">
              <position x="750" y="150"/>
              <connectionPointIn>
                <relPosition x="0" y="20"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="60"/>
                <connection refLocalId="5">
                  <position x="750" y="210"/>
                  <position x="690" y="210"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="100"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="140"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="180"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="220"/>
                <connection refLocalId="41">
                  <position x="750" y="370"/>
                  <position x="690" y="370"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="260"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="300"/>
                <connection refLocalId="18">
                  <position x="750" y="450"/>
                  <position x="690" y="450"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="340"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="380"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="420"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="460"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="500"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="540"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="580"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="620"/>
                <connection refLocalId="17">
                  <position x="750" y="770"/>
                  <position x="660" y="770"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="660"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="700"/>
                <connection refLocalId="10">
                  <position x="750" y="850"/>
                  <position x="660" y="850"/>
                </connection>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="740"/>
              </connectionPointIn>
              <connectionPointIn>
                <relPosition x="0" y="780"/>
              </connectionPointIn>
            </rightPowerRail>
            <contact localId="3" negated="false" edge="rising" width="30" height="20">
              <position x="190" y="200"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="190" y="210"/>
                  <position x="110" y="210"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Control_button</variable>
            </contact>
            <block localId="4" typeName="CTU" instanceName="CTU0" width="50" height="80">
              <position x="470" y="180"/>
              <inputVariables>
                <variable formalParameter="CU" edge="rising">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="3">
                      <position x="470" y="210"/>
                      <position x="220" y="210"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="R">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="7">
                      <position x="470" y="230"/>
                      <position x="440" y="230"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="PV">
                  <connectionPointIn>
                    <relPosition x="0" y="70"/>
                    <connection refLocalId="16">
                      <position x="470" y="250"/>
                      <position x="455" y="250"/>
                      <position x="455" y="260"/>
                      <position x="430" y="260"/>
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
                <variable formalParameter="CV">
                  <connectionPointOut>
                    <relPosition x="50" y="50"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
            <coil localId="5" negated="false" width="30" height="20">
              <position x="660" y="200"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="4" formalParameter="Q">
                  <position x="660" y="210"/>
                  <position x="520" y="210"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Reset_state</variable>
            </coil>
            <outVariable localId="6" width="62" height="20" negated="false">
              <position x="550" y="220"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="4" formalParameter="CV">
                  <position x="550" y="230"/>
                  <position x="520" y="230"/>
                </connection>
              </connectionPointIn>
              <expression>Light_bright</expression>
            </outVariable>
            <inVariable localId="7" width="70" height="20" negated="false">
              <position x="370" y="220"/>
              <connectionPointOut>
                <relPosition x="70" y="10"/>
              </connectionPointOut>
              <expression>Reset_state</expression>
            </inVariable>
            <block localId="9" typeName="TP" instanceName="TP0" width="50" height="60">
              <position x="500" y="820"/>
              <inputVariables>
                <variable formalParameter="IN">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="39" formalParameter="Q">
                      <position x="500" y="850"/>
                      <position x="463" y="850"/>
                      <position x="463" y="770"/>
                      <position x="410" y="770"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="PT">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="15">
                      <position x="500" y="870"/>
                      <position x="470" y="870"/>
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
            <coil localId="10" negated="false" width="30" height="20">
              <position x="630" y="840"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="9" formalParameter="Q">
                  <position x="630" y="850"/>
                  <position x="550" y="850"/>
                </connection>
                <connection refLocalId="43">
                  <position x="630" y="850"/>
                  <position x="600" y="850"/>
                  <position x="600" y="930"/>
                  <position x="230" y="930"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Light_output</variable>
            </coil>
            <contact localId="12" negated="true" width="30" height="20">
              <position x="270" y="760"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="42">
                  <position x="270" y="770"/>
                  <position x="200" y="770"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Flag_cicle</variable>
            </contact>
            <inVariable localId="13" width="62" height="20" negated="false">
              <position x="160" y="360"/>
              <connectionPointOut>
                <relPosition x="62" y="10"/>
              </connectionPointOut>
              <expression>Light_bright</expression>
            </inVariable>
            <inVariable localId="14" width="20" height="20" negated="false">
              <position x="200" y="390"/>
              <connectionPointOut>
                <relPosition x="20" y="10"/>
              </connectionPointOut>
              <expression>0</expression>
            </inVariable>
            <inVariable localId="15" width="80" height="20" negated="false">
              <position x="390" y="860"/>
              <connectionPointOut>
                <relPosition x="80" y="10"/>
              </connectionPointOut>
              <expression>Pulse_regulator</expression>
            </inVariable>
            <inVariable localId="16" width="20" height="20" negated="false">
              <position x="410" y="250"/>
              <connectionPointOut>
                <relPosition x="20" y="10"/>
              </connectionPointOut>
              <expression>4</expression>
            </inVariable>
            <inVariable localId="8" width="80" height="20" negated="false" executionOrderId="0">
              <position x="240" y="810"/>
              <connectionPointOut>
                <relPosition x="80" y="10"/>
              </connectionPointOut>
              <expression>T#10ms</expression>
            </inVariable>
            <inVariable localId="11" width="62" height="20" negated="false" executionOrderId="0">
              <position x="160" y="440"/>
              <connectionPointOut>
                <relPosition x="62" y="10"/>
              </connectionPointOut>
              <expression>Light_bright</expression>
            </inVariable>
            <inVariable localId="22" width="20" height="20" negated="false" executionOrderId="0">
              <position x="200" y="470"/>
              <connectionPointOut>
                <relPosition x="20" y="10"/>
              </connectionPointOut>
              <expression>1</expression>
            </inVariable>
            <block localId="26" typeName="EQ" width="60" height="60" executionOrderId="0">
              <position x="270" y="420"/>
              <inputVariables>
                <variable formalParameter="IN1">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="11">
                      <position x="270" y="450"/>
                      <position x="222" y="450"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="IN2">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="22">
                      <position x="270" y="470"/>
                      <position x="245" y="470"/>
                      <position x="245" y="480"/>
                      <position x="220" y="480"/>
                    </connection>
                  </connectionPointIn>
                </variable>
              </inputVariables>
              <inOutVariables/>
              <outputVariables>
                <variable formalParameter="OUT">
                  <connectionPointOut>
                    <relPosition x="60" y="30"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
            <inVariable localId="27" width="62" height="20" negated="false" executionOrderId="0">
              <position x="160" y="520"/>
              <connectionPointOut>
                <relPosition x="62" y="10"/>
              </connectionPointOut>
              <expression>Light_bright</expression>
            </inVariable>
            <inVariable localId="28" width="20" height="20" negated="false" executionOrderId="0">
              <position x="200" y="550"/>
              <connectionPointOut>
                <relPosition x="20" y="10"/>
              </connectionPointOut>
              <expression>2</expression>
            </inVariable>
            <block localId="29" typeName="MOVE" width="60" height="60" executionOrderId="0">
              <position x="530" y="500"/>
              <inputVariables>
                <variable formalParameter="EN">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="32" formalParameter="OUT">
                      <position x="530" y="530"/>
                      <position x="330" y="530"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="IN">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="30">
                      <position x="530" y="550"/>
                      <position x="510" y="550"/>
                      <position x="510" y="560"/>
                      <position x="490" y="560"/>
                    </connection>
                  </connectionPointIn>
                </variable>
              </inputVariables>
              <inOutVariables/>
              <outputVariables>
                <variable formalParameter="ENO">
                  <connectionPointOut>
                    <relPosition x="60" y="30"/>
                  </connectionPointOut>
                </variable>
                <variable formalParameter="OUT">
                  <connectionPointOut>
                    <relPosition x="60" y="50"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
            <inVariable localId="30" width="50" height="20" negated="false" executionOrderId="0">
              <position x="440" y="550"/>
              <connectionPointOut>
                <relPosition x="50" y="10"/>
              </connectionPointOut>
              <expression>T#5ms</expression>
            </inVariable>
            <outVariable localId="31" width="80" height="20" negated="false" executionOrderId="0">
              <position x="630" y="550"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="29" formalParameter="OUT">
                  <position x="630" y="560"/>
                  <position x="610" y="560"/>
                  <position x="610" y="550"/>
                  <position x="590" y="550"/>
                </connection>
              </connectionPointIn>
              <expression>Pulse_regulator</expression>
            </outVariable>
            <block localId="32" typeName="EQ" width="60" height="60" executionOrderId="0">
              <position x="270" y="500"/>
              <inputVariables>
                <variable formalParameter="IN1">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="27">
                      <position x="270" y="530"/>
                      <position x="222" y="530"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="IN2">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="28">
                      <position x="270" y="550"/>
                      <position x="245" y="550"/>
                      <position x="245" y="560"/>
                      <position x="220" y="560"/>
                    </connection>
                  </connectionPointIn>
                </variable>
              </inputVariables>
              <inOutVariables/>
              <outputVariables>
                <variable formalParameter="OUT">
                  <connectionPointOut>
                    <relPosition x="60" y="30"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
            <inVariable localId="33" width="62" height="20" negated="false" executionOrderId="0">
              <position x="160" y="600"/>
              <connectionPointOut>
                <relPosition x="62" y="10"/>
              </connectionPointOut>
              <expression>Light_bright</expression>
            </inVariable>
            <inVariable localId="34" width="20" height="20" negated="false" executionOrderId="0">
              <position x="200" y="630"/>
              <connectionPointOut>
                <relPosition x="20" y="10"/>
              </connectionPointOut>
              <expression>3</expression>
            </inVariable>
            <block localId="35" typeName="MOVE" width="60" height="60" executionOrderId="0">
              <position x="530" y="580"/>
              <inputVariables>
                <variable formalParameter="EN">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="38" formalParameter="OUT">
                      <position x="530" y="610"/>
                      <position x="330" y="610"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="IN">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="36">
                      <position x="530" y="630"/>
                      <position x="510" y="630"/>
                      <position x="510" y="640"/>
                      <position x="490" y="640"/>
                    </connection>
                  </connectionPointIn>
                </variable>
              </inputVariables>
              <inOutVariables/>
              <outputVariables>
                <variable formalParameter="ENO">
                  <connectionPointOut>
                    <relPosition x="60" y="30"/>
                  </connectionPointOut>
                </variable>
                <variable formalParameter="OUT">
                  <connectionPointOut>
                    <relPosition x="60" y="50"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
            <inVariable localId="36" width="50" height="20" negated="false" executionOrderId="0">
              <position x="440" y="630"/>
              <connectionPointOut>
                <relPosition x="50" y="10"/>
              </connectionPointOut>
              <expression>T#2ms</expression>
            </inVariable>
            <outVariable localId="37" width="80" height="20" negated="false" executionOrderId="0">
              <position x="630" y="630"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="35" formalParameter="OUT">
                  <position x="630" y="640"/>
                  <position x="610" y="640"/>
                  <position x="610" y="630"/>
                  <position x="590" y="630"/>
                </connection>
              </connectionPointIn>
              <expression>Pulse_regulator</expression>
            </outVariable>
            <block localId="38" typeName="EQ" width="60" height="60" executionOrderId="0">
              <position x="270" y="580"/>
              <inputVariables>
                <variable formalParameter="IN1">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="33">
                      <position x="270" y="610"/>
                      <position x="222" y="610"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="IN2">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="34">
                      <position x="270" y="630"/>
                      <position x="245" y="630"/>
                      <position x="245" y="640"/>
                      <position x="220" y="640"/>
                    </connection>
                  </connectionPointIn>
                </variable>
              </inputVariables>
              <inOutVariables/>
              <outputVariables>
                <variable formalParameter="OUT">
                  <connectionPointOut>
                    <relPosition x="60" y="30"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
            <block localId="39" typeName="TOF" instanceName="TOF0" width="50" height="60">
              <position x="360" y="740"/>
              <inputVariables>
                <variable formalParameter="IN">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="12">
                      <position x="360" y="770"/>
                      <position x="300" y="770"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="PT">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="8">
                      <position x="360" y="790"/>
                      <position x="330" y="790"/>
                      <position x="330" y="820"/>
                      <position x="320" y="820"/>
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
            <coil localId="17" negated="false" width="30" height="20" executionOrderId="0">
              <position x="630" y="760"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="39" formalParameter="Q">
                  <position x="630" y="770"/>
                  <position x="410" y="770"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Flag_cicle</variable>
            </coil>
            <block localId="40" typeName="GT" width="60" height="60">
              <position x="270" y="340"/>
              <inputVariables>
                <variable formalParameter="IN1">
                  <connectionPointIn>
                    <relPosition x="0" y="30"/>
                    <connection refLocalId="13">
                      <position x="270" y="370"/>
                      <position x="222" y="370"/>
                    </connection>
                  </connectionPointIn>
                </variable>
                <variable formalParameter="IN2">
                  <connectionPointIn>
                    <relPosition x="0" y="50"/>
                    <connection refLocalId="14">
                      <position x="270" y="390"/>
                      <position x="245" y="390"/>
                      <position x="245" y="400"/>
                      <position x="220" y="400"/>
                    </connection>
                  </connectionPointIn>
                </variable>
              </inputVariables>
              <inOutVariables/>
              <outputVariables>
                <variable formalParameter="OUT">
                  <connectionPointOut>
                    <relPosition x="60" y="30"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
            <coil localId="41" negated="false" width="30" height="20">
              <position x="660" y="360"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="40" formalParameter="OUT">
                  <position x="660" y="370"/>
                  <position x="330" y="370"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Light_on_state</variable>
            </coil>
            <contact localId="42" negated="false" width="30" height="20">
              <position x="170" y="760"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="170" y="770"/>
                  <position x="110" y="770"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Light_on_state</variable>
            </contact>
            <coil localId="18" negated="false" width="30" height="20" executionOrderId="0">
              <position x="660" y="440"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="26" formalParameter="OUT">
                  <position x="660" y="450"/>
                  <position x="330" y="450"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Full_bright</variable>
            </coil>
            <contact localId="43" negated="false" width="30" height="20">
              <position x="200" y="920"/>
              <connectionPointIn>
                <relPosition x="0" y="10"/>
                <connection refLocalId="1">
                  <position x="200" y="930"/>
                  <position x="110" y="930"/>
                </connection>
              </connectionPointIn>
              <connectionPointOut>
                <relPosition x="30" y="10"/>
              </connectionPointOut>
              <variable>Full_bright</variable>
            </contact>
            <comment localId="44" height="40" width="490">
              <position x="120" y="120"/>
              <content>
                <xhtml:p><![CDATA[Positive pulse of Control button increase de value of the state, when the value is 4 the count is reseted]]></xhtml:p>
              </content>
            </comment>
            <comment localId="45" height="40" width="480">
              <position x="130" y="290"/>
              <content>
                <xhtml:p><![CDATA[Compares the states and active diferent flags or change values according to the state behaviour]]></xhtml:p>
              </content>
            </comment>
            <comment localId="46" height="50" width="600">
              <position x="130" y="670"/>
              <content>
                <xhtml:p><![CDATA[If flags indicates tha light is turned on, TOF0 and TP0 generate a PWM signal with periode of 10ms, full_bright flag makes full bright by turning on the light permanently ]]></xhtml:p>
              </content>
            </comment>
            <comment localId="47" height="90" width="470">
              <position x="120" y="10"/>
              <content>
                <xhtml:p><![CDATA[/*********PINOUT CONFIGURATION********
Digital In:  DI0, DI1, DI2, DI3                         (%IX0.0 - %IX0.3)
Digital Out: DO0, DO1, DO2, DO3, DO4, DO5               (%QX0.0 - %QX0.5)
Analog In:   AI0, AI1, AI2, AI3, AI4, AI5               (%IW0 - %IW5)
Analog Out:  DO6, DO7                                   (%QW0 - %QW1)
***********************/]]></xhtml:p>
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
            <pouInstance name="instance0" typeName="Dimmer"/>
          </task>
        </resource>
      </configuration>
    </configurations>
  </instances>
</project>
