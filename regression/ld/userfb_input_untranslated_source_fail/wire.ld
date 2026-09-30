<?xml version="1.0" encoding="UTF-8"?>
<!-- #7580: p0.A is wired to the output of bad0, whose body is malformed
     (an IF with no END_IF) and is not translated, so that output takes any
     value each scan. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-28"/>
  <contentHeader name="UNTRANSLATED"/>
  <types/><instances/>
  <pous>
    <pou name="BAD" pouType="functionBlock">
      <interface>
        <inputVars><variable name="N"><type><INT/></type></variable></inputVars>
        <outputVars><variable name="OUT"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
IF N > 0 THEN
  OUT := TRUE;
]]></ST></body>
    </pou>
    <pou name="PASS" pouType="functionBlock">
      <interface>
        <inputVars><variable name="A"><type><BOOL/></type></variable></inputVars>
        <outputVars><variable name="Q"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
Q := A;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Y"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <block localId="10" typeName="BAD" instanceName="bad0"/>
        <block localId="11" typeName="PASS" instanceName="p0">
          <inputVariables>
            <variable formalParameter="A"><connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <outVariable localId="20"><expression>Y</expression>
          <connectionPointIn><connection refLocalId="11" formalParameter="Q"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
