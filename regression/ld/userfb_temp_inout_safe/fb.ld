<?xml version="1.0" encoding="UTF-8"?>
<!-- #7579: BOOL variables declared as VAR_TEMP and VAR_IN_OUT keep their type,
     so Boolean operators on them translate. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-28"/>
  <contentHeader name="TEMPIO"/>
  <types/><instances/>
  <pous>
    <pou name="GATE" pouType="functionBlock">
      <interface>
        <inputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </inputVars>
        <inOutVars><variable name="IO"><type><BOOL/></type></variable></inOutVars>
        <outputVars><variable name="OUT"><type><BOOL/></type></variable></outputVars>
        <tempVars><variable name="T"><type><BOOL/></type></variable></tempVars>
      </interface>
      <body><ST><![CDATA[
T := A AND B;
OUT := T AND IO;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Q"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <block localId="10" typeName="GATE" instanceName="g0"/>
        <outVariable localId="11"><expression>Q</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
