<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="regression" productVersion="1" creationDateTime="2026-09-27T00:00:00"/>
  <contentHeader name="shared"><coordinateInfo><fbd><scaling x="10" y="10"/></fbd><ld><scaling x="10" y="10"/></ld><sfc><scaling x="10" y="10"/></sfc></coordinateInfo></contentHeader>
  <types><dataTypes/><pous>
    <pou name="shared" pouType="program">
      <interface><localVars>
        <variable name="s" address="%IX0.0"><type><BOOL/></type></variable>
        <variable name="v" address="%QX0.0"><type><BOOL/></type></variable>
        <variable name="q" address="%QX0.1"><type><BOOL/></type></variable>
      </localVars></interface>
      <body><LD>
        <leftPowerRail localId="1" width="10" height="80">
          <position x="10" y="10"/>
          <connectionPointOut formalParameter="none"><relPosition x="10" y="10"/></connectionPointOut>
          <connectionPointOut formalParameter="none"><relPosition x="10" y="50"/></connectionPointOut>
        </leftPowerRail>
        <contact localId="3" width="20" height="20" negated="false">
          <position x="60" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="1"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>s</variable>
        </contact>
        <coil localId="4" width="20" height="20" negated="false" storage="set">
          <position x="140" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="3"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>v</variable>
        </coil>
        <contact localId="5" width="20" height="20" negated="false">
          <position x="60" y="50"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="1"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>v</variable>
        </contact>
        <coil localId="6" width="20" height="20" negated="false" storage="reset">
          <position x="140" y="50"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="5"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>v</variable>
        </coil>
        <block localId="7" typeName="TON" instanceName="TON0" width="50" height="60">
          <position x="140" y="80"/>
          <inputVariables>
            <variable formalParameter="IN"><connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="5"/></connectionPointIn></variable>
            <variable formalParameter="PT"><connectionPointIn><relPosition x="0" y="30"/><connection refLocalId="8"/></connectionPointIn></variable>
          </inputVariables>
          <inOutVariables/>
          <outputVariables>
            <variable formalParameter="Q"><connectionPointOut><relPosition x="50" y="10"/></connectionPointOut></variable>
            <variable formalParameter="ET"><connectionPointOut><relPosition x="50" y="30"/></connectionPointOut></variable>
          </outputVariables>
        </block>
        <inVariable localId="8" width="40" height="20" negated="false"><position x="100" y="110"/><connectionPointOut><relPosition x="40" y="10"/></connectionPointOut><expression>T#20ms</expression></inVariable>
        <rightPowerRail localId="2" width="10" height="80">
          <position x="220" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="4"/></connectionPointIn>
          <connectionPointIn><relPosition x="0" y="50"/><connection refLocalId="6"/></connectionPointIn>
          
        </rightPowerRail>
      </LD></body>
    </pou>
  </pous></types>
  <instances><configurations><configuration name="c"><resource name="r">
    <task name="main" interval="T#10ms" priority="0"><pouInstance name="p" typeName="shared"/></task>
  </resource></configuration></configurations></instances>
</project>
