<?xml version='1.0' encoding='utf-8'?>
<!-- m is set, reset, then set again in rung order: in a scan where several of
     its coils are powered, the last powered one wins, whichever kind it is. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="regression" productVersion="1" creationDateTime="2026-09-29T00:00:00"/>
  <contentHeader name="latch"><coordinateInfo><ld><scaling x="10" y="10"/></ld></coordinateInfo></contentHeader>
  <types><dataTypes/><pous>
    <pou name="latch" pouType="program">
      <interface>
        <inputVars>
          <variable name="a"><type><BOOL/></type></variable>
          <variable name="b"><type><BOOL/></type></variable>
          <variable name="c"><type><BOOL/></type></variable>
        </inputVars>
        <localVars><variable name="m"><type><BOOL/></type></variable></localVars>
      </interface>
      <body><LD>
        <rung localId="1"><contact variable="a"/><coil variable="m" kind="set"/></rung>
        <rung localId="2"><contact variable="b"/><coil variable="m" kind="reset"/></rung>
        <rung localId="3"><contact variable="c"/><coil variable="m" kind="set"/></rung>
      </LD></body>
    </pou>
  </pous></types>
  <instances><configurations><configuration name="c"><resource name="r">
    <task name="main" interval="T#10ms" priority="0"><pouInstance name="p" typeName="latch"/></task>
  </resource></configuration></configurations></instances>
</project>
