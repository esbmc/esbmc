<?xml version="1.0" encoding="UTF-8"?>
<!-- A contact before a TON in a textual rung: the block reads IN from its named
     variable, so the contact would not gate it. The front end refuses the rung. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-18"/>
  <contentHeader name="Gated"/>
  <types/><instances/>
  <pous>
    <pou name="Gated" pouType="program">
      <interface>
        <inputVars>
          <variable name="Enable"><type><BOOL/></type></variable>
          <variable name="Start"><type><BOOL/></type></variable>
        </inputVars>
        <localVars>
          <variable name="PT"><type><TIME/></type><initialValue><simpleValue value="T#3ms"/></initialValue></variable>
          <variable name="ET"><type><TIME/></type></variable>
        </localVars>
        <outputVars>
          <variable name="Done"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact><variable>Enable</variable></contact><block typeName="TON" instanceName="t1"><variable formalParameter="IN">Start</variable><variable formalParameter="PT">PT</variable><variable formalParameter="Q">Done</variable><variable formalParameter="ET">ET</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
