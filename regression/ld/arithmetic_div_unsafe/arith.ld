<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-10"/>
  <contentHeader name="Arith"/>
  <types/>
  <instances/>
  <pous>
    <pou name="Arith" pouType="program">
      <interface>
        <inputVars>
          <variable name="A"><type><INT/></type></variable>
          <variable name="B"><type><INT/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Sum"><type><INT/></type></variable>
          <variable name="Diff"><type><INT/></type></variable>
          <variable name="Prod"><type><INT/></type></variable>
          <variable name="Quot"><type><INT/></type></variable>
          <variable name="Mov"><type><INT/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="ADD" instanceName="a1"><variable formalParameter="IN1">A</variable><variable formalParameter="IN2">B</variable><variable formalParameter="OUT">Sum</variable></block></rung>
          <rung localId="2"><block typeName="SUB" instanceName="s1"><variable formalParameter="IN1">A</variable><variable formalParameter="IN2">B</variable><variable formalParameter="OUT">Diff</variable></block></rung>
          <rung localId="3"><block typeName="MUL" instanceName="m1"><variable formalParameter="IN1">A</variable><variable formalParameter="IN2">B</variable><variable formalParameter="OUT">Prod</variable></block></rung>
          <rung localId="4"><block typeName="DIV" instanceName="d1"><variable formalParameter="IN1">A</variable><variable formalParameter="IN2">B</variable><variable formalParameter="OUT">Quot</variable></block></rung>
          <rung localId="5"><block typeName="MOVE" instanceName="mv1"><variable formalParameter="IN1">A</variable><variable formalParameter="OUT">Mov</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
