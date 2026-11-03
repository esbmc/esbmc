<?xml version='1.0' encoding='utf-8'?>
<!-- #8211: an edge contact compares with its operand at its own previous evaluation. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="edge" productVersion="1" creationDateTime="2026-10-20T00:00:00"/>
  <contentHeader name="edge"><coordinateInfo><ld><scaling x="10" y="10"/></ld></coordinateInfo></contentHeader>
  <types><dataTypes/><pous>
    <pou name="edge" pouType="program">
      <interface><localVars>
        <variable name="go" address="%IX0.0"><type><BOOL/></type></variable>
        <variable name="v"><type><BOOL/></type></variable>
        <variable name="hit" address="%QX0.0"><type><BOOL/></type></variable>
      </localVars></interface>
      <body><LD>
        <leftPowerRail localId="1" width="10" height="80">
          <position x="10" y="10"/>
          <connectionPointOut formalParameter=""><relPosition x="10" y="10"/></connectionPointOut>
          <connectionPointOut formalParameter=""><relPosition x="10" y="50"/></connectionPointOut>
        </leftPowerRail>
        <contact localId="3" width="20" height="20" negated="false" edge="rising" executionOrderId="1">
          <position x="60" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="1"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>v</variable>
        </contact>
        <coil localId="4" width="20" height="20" negated="false" executionOrderId="2">
          <position x="200" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="3"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>hit</variable>
        </coil>
        <contact localId="5" width="20" height="20" negated="false" executionOrderId="3">
          <position x="60" y="50"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="1"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>go</variable>
        </contact>
        <coil localId="6" width="20" height="20" negated="false" executionOrderId="4">
          <position x="200" y="50"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="5"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>v</variable>
        </coil>
        <rightPowerRail localId="2" width="10" height="80">
          <position x="250" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="4"/></connectionPointIn>
          <connectionPointIn><relPosition x="0" y="50"/><connection refLocalId="6"/></connectionPointIn>
        </rightPowerRail>
      </LD></body>
    </pou>
  </pous></types>
  <instances><configurations><configuration name="c"><resource name="r">
    <task name="main" interval="T#10ms" priority="0"><pouInstance name="p" typeName="edge"/></task>
  </resource></configuration></configurations></instances>
</project>
