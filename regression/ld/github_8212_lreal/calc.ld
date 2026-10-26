<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-20"/>
  <contentHeader name="CALC"/>
  <types/><instances/>
  <pous>
    <pou name="CALC" pouType="functionBlock">
      <interface>
        <outputVars>
          <variable name="OUT"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="a"><type><LREAL/></type></variable>
          <variable name="b"><type><LREAL/></type></variable>
        </localVars>
      </interface>
      <body>
        <ST><![CDATA[
a := 16777216.0;
b := 1.0;
OUT := a + b <> a;
]]></ST>
      </body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface>
        <outputVars>
          <variable name="Safe"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <block localId="10" typeName="CALC" instanceName="c0"/>
          <outVariable localId="11">
            <expression>Safe</expression>
            <connectionPointIn>
              <connection refLocalId="10" formalParameter="OUT"/>
            </connectionPointIn>
          </outVariable>
        </LD>
      </body>
    </pou>
  </pous>
</project>
