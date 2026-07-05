<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-10"/>
  <contentHeader name="FB"/>
  <types/><instances/>
  <pous>
    <pou name="FB" pouType="program">
      <interface>
        <inputVars>
          <variable name="Gate"><type><BOOL/></type></variable>
          <variable name="Run"><type><BOOL/></type></variable>
          <variable name="Up"><type><BOOL/></type></variable>
          <variable name="Dn"><type><BOOL/></type></variable>
          <variable name="Rst"><type><BOOL/></type></variable>
          <variable name="PT"><type><INT/></type></variable>
          <variable name="PV"><type><INT/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Open"><type><BOOL/></type></variable>
          <variable name="TQ"><type><BOOL/></type></variable>
          <variable name="CUQ"><type><BOOL/></type></variable>
          <variable name="CDQ"><type><BOOL/></type></variable>
          <variable name="ET"><type><INT/></type></variable>
          <variable name="CVu"><type><INT/></type></variable>
          <variable name="CVd"><type><INT/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="Gate"/><coil variable="Open"/></rung>
          <rung localId="2"><block typeName="TON" instanceName="t1"><variable formalParameter="IN">Run</variable><variable formalParameter="PT">PT</variable><variable formalParameter="Q">TQ</variable><variable formalParameter="ET">ET</variable></block></rung>
          <rung localId="3"><block typeName="CTU" instanceName="cu1"><variable formalParameter="CU">Up</variable><variable formalParameter="R">Rst</variable><variable formalParameter="PV">PV</variable><variable formalParameter="Q">CUQ</variable><variable formalParameter="CV">CVu</variable></block></rung>
          <rung localId="4"><block typeName="CTD" instanceName="cd1"><variable formalParameter="CD">Dn</variable><variable formalParameter="PV">PV</variable><variable formalParameter="Q">CDQ</variable><variable formalParameter="CV">CVd</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
