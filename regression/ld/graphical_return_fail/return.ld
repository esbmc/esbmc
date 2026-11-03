<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-19"/>
  <contentHeader name="P"/>
  <types><dataTypes/><pous>
    <pou name="P" pouType="program">
      <!-- The return ends the body before the rung that sets y (IEC 61131-3:2025 8.1.6), so y stays FALSE; the front end refuses it. -->
      <interface>
        <inputVars><variable name="a"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="y"><type><BOOL/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <leftPowerRail localId="1"><position x="0" y="0"/><connectionPointOut/></leftPowerRail>
          <return localId="2"><position x="50" y="10"/>
            <connectionPointIn><connection refLocalId="1"/></connectionPointIn></return>
          <contact localId="3" negated="false" edge="none"><position x="20" y="40"/>
            <connectionPointIn><connection refLocalId="1"/></connectionPointIn><connectionPointOut/><variable>a</variable></contact>
          <coil localId="4" negated="false" storage="none"><position x="150" y="40"/>
            <connectionPointIn><connection refLocalId="3"/></connectionPointIn><connectionPointOut/><variable>y</variable></coil>
        </LD>
      </body>
    </pou>
  </pous></types>
  <instances/>
</project>
