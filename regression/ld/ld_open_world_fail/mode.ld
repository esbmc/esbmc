<?xml version='1.0' encoding='utf-8'?>
<!-- Twin of ld_closed_world_safe without --ld-closed-world: an HMI may set the
     %M variable mode between scans, so q can be on in two consecutive scans. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="regression" productVersion="1" creationDateTime="2026-09-29T00:00:00"/>
  <contentHeader name="p"><coordinateInfo><ld><scaling x="10" y="10"/></ld></coordinateInfo></contentHeader>
  <types><dataTypes/><pous>
    <pou name="p" pouType="program">
      <interface>
        <localVars>
          <variable name="mode" address="%MX0.0"><type><BOOL/></type></variable>
          <variable name="r"><type><BOOL/></type></variable>
        </localVars>
        <outputVars><variable name="q"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><LD>
        <rung localId="1"><contact variable="q"/><coil variable="r"/></rung>
        <rung localId="2"><contact variable="mode"/><coil variable="q"/></rung>
        <rung localId="3"><contact variable="q"/><coil variable="mode" kind="reset"/></rung>
      </LD></body>
    </pou>
  </pous></types>
  <instances><configurations><configuration name="c"><resource name="r0">
    <task name="main" interval="T#10ms" priority="0"><pouInstance name="i" typeName="p"/></task>
  </resource></configuration></configurations></instances>
</project>
