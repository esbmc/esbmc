<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-20"/>
  <contentHeader name="INIT"/>
  <types/><instances/>
  <pous>
    <pou name="CHECK" pouType="functionBlock">
      <interface>
        <inputVars>
          <variable name="IN"><type><REAL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="OUT"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <ST><![CDATA[
OUT := IN = 16777216.0;
]]></ST>
      </body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface>
        <localVars>
          <variable name="R"><type><REAL/></type><initialValue><simpleValue value="16777217"/></initialValue></variable>
        </localVars>
        <outputVars>
          <variable name="Safe"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><LD>
        <inVariable localId="1"><expression>R</expression></inVariable>
        <block localId="10" typeName="CHECK" instanceName="c0">
          <inputVariables>
            <variable formalParameter="IN"><connectionPointIn><connection refLocalId="1"/></connectionPointIn></variable>
          </inputVariables>
          <outputVariables>
            <variable formalParameter="OUT"><connectionPointOut/></variable>
          </outputVariables>
        </block>
        <outVariable localId="20"><expression>Safe</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
