<?xml version="1.0" encoding="UTF-8"?>
<!-- #8210: an EN wired to the left power rail is always true and is accepted; ENO ends at the right rail. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-18"/>
  <contentHeader name="RELAY"/>
  <types/><instances/>
  <pous>
    <pou name="RELAY" pouType="functionBlock">
      <interface>
        <inputVars><variable name="IN1"><type><BOOL/></type></variable></inputVars>
        <outputVars><variable name="OUT"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
OUT := IN1;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface>
        <inputVars>
          <variable name="ENABLE"><type><BOOL/></type></variable>
          <variable name="GO"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Y"><type><BOOL/></type></variable>
          <variable name="OK"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><LD>
        <leftPowerRail localId="0"><connectionPointOut formalParameter="none"/></leftPowerRail>
        <inVariable localId="1"><expression>ENABLE</expression></inVariable>
        <inVariable localId="2"><expression>GO</expression></inVariable>
        <block localId="10" typeName="RELAY" instanceName="relay0">
          <inputVariables>
            <variable formalParameter="EN"><connectionPointIn><connection refLocalId="0"/></connectionPointIn></variable>
            <variable formalParameter="IN1"><connectionPointIn><connection refLocalId="2"/></connectionPointIn></variable>
          </inputVariables>
          <outputVariables>
            <variable formalParameter="ENO"><connectionPointOut/></variable>
            <variable formalParameter="OUT"><connectionPointOut/></variable>
          </outputVariables>
        </block>
        <outVariable localId="20"><expression>Y</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></outVariable>
        <rightPowerRail localId="30"><connectionPointIn><connection refLocalId="10" formalParameter="ENO"/></connectionPointIn></rightPowerRail>
      </LD></body>
    </pou>
  </pous>
</project>
