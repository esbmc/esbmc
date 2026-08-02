<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-26"/>
  <contentHeader name="CALC"/>
  <types/><instances/>
  <pous>
    <pou name="CALC" pouType="functionBlock">
      <interface>
        <outputVars>
          <variable name="OUT"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="x"><type><REAL/></type></variable>
        </localVars>
      </interface>
      <body>
        <ST><![CDATA[
x := 1.0 + 2.0 * 3.0;
OUT := x = 9.0;
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
