<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-04"/>
  <contentHeader name="Timers"/>
  <types/><instances/>
  <pous>
    <pou name="Timers" pouType="program">
      <interface>
        <inputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </inputVars>
        <localVars>
          <variable name="H1"><type><BOOL/></type></variable>
          <variable name="H2"><type><BOOL/></type></variable>
          <variable name="QA"><type><BOOL/></type></variable>
          <variable name="QB"><type><BOOL/></type></variable>
          <variable name="ETA"><type><INT/></type></variable>
          <variable name="ETB"><type><INT/></type></variable>
          <variable name="PT0"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="H1"/><coil variable="H2"/></rung>
          <rung localId="2"><contact variable="B"/><coil variable="H1"/></rung>
          <rung localId="3"><block typeName="TON"><variable formalParameter="IN">A</variable><variable formalParameter="PT">PT0</variable><variable formalParameter="Q">A</variable><variable formalParameter="ET">ETA</variable></block></rung>
          <rung localId="4"><block typeName="TON"><variable formalParameter="IN">B</variable><variable formalParameter="PT">PT0</variable><variable formalParameter="Q">QB</variable><variable formalParameter="ET">ETB</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
