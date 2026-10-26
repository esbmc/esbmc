<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-18"/>
  <contentHeader name="P"/>
  <types><dataTypes/><pous>
    <pou name="GATE" pouType="functionBlock">
      <interface>
        <inputVars><variable name="IN"><type><BOOL/></type></variable></inputVars>
        <outputVars><variable name="OUT"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
OUT := IN;
]]></ST></body>
    </pou>
    <pou name="P" pouType="program">
      <!-- A user FB in power flow. The coil would read G1__Q, which the block never writes;
           declaring that name made the earlier front end verify the wrong variable. -->
      <interface>
        <inputVars><variable name="a"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="y"><type><BOOL/></type></variable><variable name="G1__Q"><type><BOOL/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <leftPowerRail localId="1"><position x="0" y="0"/><connectionPointOut/></leftPowerRail>
          <contact localId="2" negated="false" edge="none"><position x="20" y="10"/>
            <connectionPointIn><connection refLocalId="1"/></connectionPointIn><connectionPointOut/><variable>a</variable></contact>
          <block localId="3" typeName="GATE" instanceName="G1"><position x="50" y="10"/>
            <inputVariables>
              <variable formalParameter="IN"><connectionPointIn><connection refLocalId="2"/></connectionPointIn></variable>
            </inputVariables>
            <outputVariables><variable formalParameter="OUT"/></outputVariables>
          </block>
          <coil localId="5" negated="false" storage="none"><position x="150" y="10"/>
            <connectionPointIn><connection refLocalId="3" formalParameter="OUT"/></connectionPointIn><connectionPointOut/><variable>y</variable></coil>
        </LD>
      </body>
    </pou>
  </pous></types>
  <instances/>
</project>
