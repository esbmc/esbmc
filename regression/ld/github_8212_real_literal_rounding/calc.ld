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
          <variable name="a"><type><REAL/></type></variable>
        </localVars>
      </interface>
      <body>
        <ST><![CDATA[
a := 1.00000017881393432617187499;
OUT := a = 1.00000011920928955078125;
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
