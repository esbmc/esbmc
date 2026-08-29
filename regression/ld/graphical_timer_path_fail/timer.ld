<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="regression" productVersion="1" creationDateTime="2026-07-30T00:00:00"/>
  <contentHeader name="timer_path"><coordinateInfo><ld><scaling x="10" y="10"/></ld></coordinateInfo></contentHeader>
  <types><dataTypes/><pous>
    <pou name="timer_path" pouType="program">
      <interface><localVars>
        <variable name="a" address="%IX0.0"><type><BOOL/></type></variable>
        <variable name="PT"><type><INT/></type><initialValue><simpleValue value="2"/></initialValue></variable>
        <variable name="q" address="%QX0.0"><type><BOOL/></type></variable>
      </localVars></interface>
      <body><LD>
        <leftPowerRail localId="1" width="10" height="40">
          <position x="10" y="10"/>
          <connectionPointOut><relPosition x="10" y="10"/></connectionPointOut>
        </leftPowerRail>
        <contact localId="3" width="20" height="20" negated="false">
          <position x="60" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="1"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>a</variable>
        </contact>
        <inVariable localId="6" width="40" height="20" negated="false">
          <position x="100" y="50"/>
          <connectionPointOut><relPosition x="40" y="10"/></connectionPointOut>
          <expression>PT</expression>
        </inVariable>
        <block localId="5" width="60" height="40" typeName="TON" instanceName="T0">
          <position x="140" y="10"/>
          <inputVariables>
            <variable formalParameter="IN">
              <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="3"/></connectionPointIn>
            </variable>
            <variable formalParameter="PT">
              <connectionPointIn><relPosition x="0" y="30"/><connection refLocalId="6"/></connectionPointIn>
            </variable>
          </inputVariables>
          <inOutVariables/>
          <outputVariables>
            <variable formalParameter="Q"><connectionPointOut><relPosition x="60" y="10"/></connectionPointOut></variable>
          </outputVariables>
        </block>
        <coil localId="4" width="20" height="20" negated="false">
          <position x="240" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="5"/></connectionPointIn>
          <connectionPointOut><relPosition x="20" y="10"/></connectionPointOut>
          <variable>q</variable>
        </coil>
        <rightPowerRail localId="2" width="10" height="40">
          <position x="300" y="10"/>
          <connectionPointIn><relPosition x="0" y="10"/><connection refLocalId="4"/></connectionPointIn>
        </rightPowerRail>
      </LD></body>
    </pou>
  </pous></types>
  <instances><configurations><configuration name="c"><resource name="r">
    <task name="main" interval="T#10ms" priority="0"><pouInstance name="p" typeName="timer_path"/></task>
  </resource></configuration></configurations></instances>
</project>
