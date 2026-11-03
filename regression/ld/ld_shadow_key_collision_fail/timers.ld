<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-12"/>
  <contentHeader name="P"/>
  <types/><instances/>
  <pous>
    <pou name="P" pouType="program">
      <!-- Two unnamed timers with one Q variable would share one previous-input shadow. -->
      <interface>
        <inputVars><variable name="a"><type><BOOL/></type></variable><variable name="b"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="done"><type><BOOL/></type></variable><variable name="E1"><type><INT/></type></variable><variable name="E2"><type><INT/></type></variable><variable name="PT"><type><INT/></type><initialValue><simpleValue value="2"/></initialValue></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="TON"><variable formalParameter="IN">a</variable><variable formalParameter="PT">PT</variable><variable formalParameter="Q">done</variable><variable formalParameter="ET">E1</variable></block></rung>
          <rung localId="2"><block typeName="TON"><variable formalParameter="IN">b</variable><variable formalParameter="PT">PT</variable><variable formalParameter="Q">done</variable><variable formalParameter="ET">E2</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
