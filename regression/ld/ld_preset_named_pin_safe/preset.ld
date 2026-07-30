<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="regression" productVersion="1" creationDateTime="2026-07-30T00:00:00"/>
  <contentHeader name="preset"><coordinateInfo><ld><scaling x="10" y="10"/></ld></coordinateInfo></contentHeader>
  <types><dataTypes/><pous>
    <pou name="preset" pouType="program">
      <interface><localVars>
        <variable name="a" address="%IX0.0"><type><BOOL/></type></variable>
        <variable name="PV"><type><INT/></type><initialValue><simpleValue value="notanumber"/></initialValue></variable>
        <variable name="q" address="%QX0.0"><type><BOOL/></type></variable>
      </localVars></interface>
      <body><LD>
        <rung localId="1"><contact variable="a"/><coil variable="q"/></rung>
      </LD></body>
    </pou>
  </pous></types>
  <instances><configurations><configuration name="c"><resource name="r">
    <task name="main" interval="T#10ms" priority="0"><pouInstance name="p" typeName="preset"/></task>
  </resource></configuration></configurations></instances>
</project>
