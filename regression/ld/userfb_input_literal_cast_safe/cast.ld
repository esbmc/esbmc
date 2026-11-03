<?xml version="1.0" encoding="UTF-8"?>
<!-- #7580: integer literals wired to BOOL, REAL and INT inputs of a user FB. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-27"/>
  <contentHeader name="CAST"/>
  <types/><instances/>
  <pous>
    <pou name="CHECK" pouType="functionBlock">
      <interface>
        <inputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="R"><type><REAL/></type></variable>
          <variable name="Z"><type><INT/></type></variable>
        </inputVars>
        <outputVars><variable name="OK"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
OK := FALSE;
IF A THEN
  IF R > 4.0 THEN
    OK := TRUE;
  END_IF;
END_IF;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Y"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <inVariable localId="1"><expression>5</expression></inVariable>
        <inVariable localId="2"><expression> 99999999999999999999 </expression></inVariable>
        <block localId="10" typeName="CHECK" instanceName="c0">
          <inputVariables>
            <variable formalParameter="A"><connectionPointIn><connection refLocalId="1"/></connectionPointIn></variable>
            <variable formalParameter="R"><connectionPointIn><connection refLocalId="1"/></connectionPointIn></variable>
            <variable formalParameter="Z"><connectionPointIn><connection refLocalId="2"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <outVariable localId="20"><expression>Y</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OK"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
