<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-12"/>
  <contentHeader name="P"/>
  <types/><instances/>
  <pous>
    <pou name="P" pouType="program">
      <!-- EN of a standard block is not modeled, so a wired EN is refused. -->
      <interface>
        <inputVars><variable name="In"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="En"><type><BOOL/></type></variable><variable name="Q1"><type><BOOL/></type></variable><variable name="ET1"><type><INT/></type></variable><variable name="PT"><type><INT/></type><initialValue><simpleValue value="2"/></initialValue></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="TON" instanceName="t1"><variable formalParameter="EN">En</variable><variable formalParameter="IN">In</variable><variable formalParameter="PT">PT</variable><variable formalParameter="Q">Q1</variable><variable formalParameter="ET">ET1</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
