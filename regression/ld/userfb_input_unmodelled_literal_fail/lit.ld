<?xml version="1.0" encoding="UTF-8"?>
<!-- #7580: user FB inputs wired to a blank inVariable and to the REAL literal 2.5. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-27"/>
  <contentHeader name="LIT"/>
  <types/><instances/>
  <pous>
    <pou name="CHECK" pouType="functionBlock">
      <interface>
        <inputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="R"><type><REAL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="OK"><type><BOOL/></type></variable>
          <variable name="SEEN"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><ST><![CDATA[
OK := R > 4.0;
SEEN := A;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Y"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <inVariable localId="1"><expression>   </expression></inVariable>
        <inVariable localId="2"><expression>2.5</expression></inVariable>
        <block localId="10" typeName="CHECK" instanceName="c0">
          <inputVariables>
            <variable formalParameter="A"><connectionPointIn><connection refLocalId="1"/></connectionPointIn></variable>
            <variable formalParameter="R"><connectionPointIn><connection refLocalId="2"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <outVariable localId="20"><expression>Y</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OK"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
