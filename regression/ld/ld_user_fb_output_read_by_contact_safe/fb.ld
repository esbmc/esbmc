<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-11"/>
  <contentHeader name="ZERO"/>
  <types/><instances/>
  <pous>
    <pou name="ZERO" pouType="functionBlock">
      <interface>
        <outputVars><variable name="OUT"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
OUT := FALSE;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface>
        <outputVars><variable name="Out"><type><BOOL/></type></variable></outputVars>
        <localVars><variable name="V"><type><BOOL/></type></variable></localVars>
      </interface>
      <body><LD>
        <leftPowerRail localId="0"><position x="0" y="0"/><connectionPointOut formalParameter="none"/></leftPowerRail>
        <contact localId="3" negated="false" storage="none" edge="none"><position x="20" y="10"/><connectionPointIn><connection refLocalId="0"/></connectionPointIn><connectionPointOut/><variable>V</variable></contact>
        <coil localId="4" negated="false" storage="none"><position x="60" y="10"/><connectionPointIn><connection refLocalId="3"/></connectionPointIn><connectionPointOut/><variable>Out</variable></coil>
        <rightPowerRail localId="2147483646"><position x="400" y="0"/><connectionPointIn><connection refLocalId="4"/></connectionPointIn></rightPowerRail>
        <block localId="10" typeName="ZERO" instanceName="z0"/>
        <outVariable localId="11"><expression>V</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
